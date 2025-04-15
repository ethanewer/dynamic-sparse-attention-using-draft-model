from typing import Callable, Optional

import torch
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
    logger,
)

from .util import compress_kv, vertical_slash_sparse_attention_forward


class LlamaAttentionSSA(LlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
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
                    self.layer_idx,
                    cache_kwargs,
                )
            else:
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs,
                )

        if indices is not None:
            v_idx = indices[..., : self.config.num_vertical].int()
            s_idx = torch.arange(
                self.config.prefill_window_size,
                -1,
                -64,
                dtype=v_idx.dtype,
                device=v_idx.device,
            )[None, None].expand(*v_idx.shape[:-1], -1)

            attn_output, attn_weights = vertical_slash_sparse_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                v_idx,
                s_idx,
            )
        else:
            attention_interface: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                if self.config._attn_implementation == "sdpa" and kwargs.get(
                    "output_attentions", False
                ):
                    logger.warning_once(  # type: ignore
                        "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                        'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                    )
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[  # type: ignore
                        self.config._attn_implementation
                    ]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights  # type: ignore


def update_llama_model_for_sparse_prefill_snapkv(model):
    model.config.window_size = 64
    model.config.max_capacity_prompt = 1024
    model.config.prefill_window_size = 1024
    model.config.num_vertical = 1024
    model.config.query_aggregation = "mean"
    model.config.pooling = "mean"
    model.config.kernel_size = 15

    for i in range(len(model.model.layers)):
        model.model.layers[i].self_attn.forward = LlamaAttentionSSA.forward.__get__(
            model.model.layers[i].self_attn,
            type(model.model.layers[i].self_attn),
        )

    return model


def reset_llama_model(model):
    for i in range(len(model.model.layers)):
        attn_layer = model.model.layers[i].self_attn
        attn_layer.forward = LlamaAttention.forward.__get__(
            attn_layer,
            type(attn_layer),
        )

    return model
