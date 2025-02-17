from typing import Callable, Optional

import torch
from transformers import Cache  # type: ignore
from transformers.models.llama.modeling_llama import (  # type: ignore
    ALL_ATTENTION_FUNCTIONS,
    LlamaAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
    logger,
    repeat_kv,
)

from .snapkv_util import init_snapkv


def undo_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_attention_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    num_key_value_heads = num_attention_heads // n_rep
    return hidden_states.view(
        batch,
        num_key_value_heads,
        n_rep,
        seq_len,
        head_dim,
    ).mean(dim=2)


class LlamaAttentionSnapKV(LlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        init_snapkv(self)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

            if past_key_value.get_seq_length() == 0:  # [SnapKV] add kv_cluster
                key_states_compress, value_states_compress = self.kv_cluster.update_kv(  # type: ignore
                    repeat_kv(key_states, self.num_key_value_groups),
                    query_states,
                    repeat_kv(value_states, self.num_key_value_groups),
                    attention_mask,
                )

                key_states_compress = undo_repeat_kv(
                    key_states_compress,
                    self.num_key_value_groups,
                )
                value_states_compress = undo_repeat_kv(
                    value_states_compress,
                    self.num_key_value_groups,
                )

                past_key_value.update(
                    key_states_compress,
                    value_states_compress,
                    self.layer_idx,
                    cache_kwargs,
                )
            else:
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

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


def update_llama_model_for_snapkv(model):
    for i in range(len(model.model.layers)):
        attn_layer = model.model.layers[i].self_attn
        attn_layer.forward = LlamaAttentionSnapKV.forward.__get__(
            attn_layer,
            type(attn_layer),
        )

    return model
