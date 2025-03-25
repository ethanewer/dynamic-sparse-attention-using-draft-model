from typing import Callable, Optional

import torch
from torch import Tensor
from transformers import Cache  # type: ignore
from transformers.models.qwen2.modeling_qwen2 import (  # type: ignore
    ALL_ATTENTION_FUNCTIONS,
    Qwen2Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
    logger,
)

from .util import das_minference_attention_forward, compress_states


class Qwen2AttentionDASMInference(Qwen2Attention):
    def forward(  # type: ignore
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[Tensor, Optional[Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        is_prefill = (
            past_key_value is None or past_key_value.get_seq_length(self.layer_idx) == 0
        )
        if is_prefill and "kwargs" in kwargs:
            assert not kwargs.get("output_attentions", False)
            assert attention_mask is None

            v_idx: Tensor = kwargs["kwargs"]["v_idx"][self.layer_idx]
            s_idx: Tensor = kwargs["kwargs"]["s_idx"][self.layer_idx]
            window_size: int = kwargs["kwargs"]["window_size"]

            if past_key_value is not None:
                past_key_value.update(
                    key_states=compress_states(key_states, v_idx, window_size),
                    value_states=compress_states(value_states, v_idx, window_size),
                    layer_idx=self.layer_idx,
                )

            attn_output, attn_weights = das_minference_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                v_idx,
                s_idx,
            )
        else:
            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {
                    "sin": sin,  # type: ignore
                    "cos": cos,  # type: ignore
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs,
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

            sliding_window = None
            if (
                self.config.use_sliding_window
                and getattr(self.config, "sliding_window", None) is not None
                and self.layer_idx >= self.config.max_window_layers
            ):
                sliding_window = self.config.sliding_window

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=sliding_window,  # main diff with Llama
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights  # type: ignore


def update_qwen2_model_for_das_minference(model):
    for i in range(len(model.model.layers)):
        attn_layer = model.model.layers[i].self_attn
        attn_layer.forward = Qwen2AttentionDASMInference.forward.__get__(
            attn_layer,
            type(attn_layer),
        )

    return model


def reset_qwen2_model(model):
    for i in range(len(model.model.layers)):
        attn_layer = model.model.layers[i].self_attn
        attn_layer.forward = Qwen2Attention.forward.__get__(
            attn_layer,
            type(attn_layer),
        )

    return model
