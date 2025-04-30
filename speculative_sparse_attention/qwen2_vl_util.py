from typing import Optional

import torch
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLFlashAttention2,
    apply_multimodal_rotary_pos_emb,
    logger,
    repeat_kv,
)

from .util import compress_kv, vertical_slash_sparse_attention_forward


class Qwen2_5_VLAttentionSSA(Qwen2_5_VLFlashAttention2):
    def forward(  # type: ignore
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(  # type: ignore
                "Qwen2_5_VLModel is using Qwen2_5_VLSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(  # type: ignore
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings  # type: ignore
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            self.rope_scaling["mrope_section"],  # type: ignore
        )

        indices = None
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

            if (
                past_key_value.get_seq_length(self.layer_idx) == 0
                and query_states.shape[2] > self.config.max_capacity_prompt
            ):
                key_states_compress, value_states_compress, indices = compress_kv(
                    key_states,
                    query_states,
                    value_states,
                    window_size=self.config.window_size,
                    max_capacity_prompt=self.config.max_capacity_prompt,
                    num_vertical=self.config.num_vertical,
                    query_aggregation=self.config.query_aggregation,
                    pooling=self.config.pooling,
                    kernel_size=self.config.kernel_size,
                )

                past_key_value.update(
                    key_states_compress,
                    value_states_compress,
                    self.layer_idx,  # type: ignore
                    cache_kwargs,
                )
            else:
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,  # type: ignore
                    cache_kwargs,
                )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        num_vertical = self.config.num_vertical
        prefill_window_size = self.config.prefill_window_size
        if indices is not None and q_len > num_vertical + prefill_window_size:
            v_idx = indices[..., :num_vertical].int()
            s_idx = torch.arange(
                prefill_window_size,
                -1,
                -64,
                dtype=v_idx.dtype,
                device=v_idx.device,
            )[None, None].expand(*v_idx.shape[:-1], -1)

            v_idx = repeat_kv(v_idx[..., None], self.num_key_value_groups)[..., 0]
            s_idx = repeat_kv(s_idx[..., None], self.num_key_value_groups)[..., 0]

            attn_output, _ = vertical_slash_sparse_attention_forward(
                None,
                query_states,
                key_states,
                value_states,
                v_idx,
                s_idx,
            )
        else:
            causal_mask = attention_mask
            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and attention_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()

            # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
            # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal = True if causal_mask is None and q_len > 1 else False

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value  # type: ignore


def update_qwen2_vl_model_for_ssa(model):
    model.config.window_size = 64
    model.config.max_capacity_prompt = 1024
    model.config.prefill_window_size = 1024
    model.config.num_vertical = 1024
    model.config.query_aggregation = "mean"
    model.config.pooling = "mean"
    model.config.kernel_size = 15

    for i in range(len(model.model.layers)):
        model.model.layers[
            i
        ].self_attn.forward = Qwen2_5_VLAttentionSSA.forward.__get__(
            model.model.layers[i].self_attn,
            type(model.model.layers[i].self_attn),
        )

    return model


def reset_qwen2_vl_model(model):
    for i in range(len(model.model.layers)):
        attn_layer = model.model.layers[i].self_attn
        attn_layer.forward = Qwen2_5_VLFlashAttention2.forward.__get__(
            attn_layer,
            type(attn_layer),
        )

    return model
