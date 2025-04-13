from typing import Callable, Optional

import torch
from torch import Tensor
from transformers import Cache  # type: ignore
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (  # type: ignore
    LlamaAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
    logger,
)

from .dynamic_attention_sinks_attention import (
    dynamic_attention_sinks_flash_attn_forward,
    dynamic_attention_sinks_spda_forward,
)
from .eager_attention_output_unnormalized import (
    eager_attention_output_unnormalized_forward,
)


class LlamaAttentionDynamicAttentionSinks(LlamaAttention):
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

        if (
            "dynamic_attention_sinks_block_size" in kwargs
            and "dynamic_attention_sinks_indices" in kwargs
            and (
                past_key_value is None
                or past_key_value.get_seq_length(self.layer_idx) == 0  # type: ignore
            )
        ):
            assert not kwargs.get("output_attentions", False)
            assert attention_mask is not None

            block_size = kwargs["dynamic_attention_sinks_block_size"]
            indices = kwargs["dynamic_attention_sinks_indices"][self.layer_idx]

            if past_key_value is not None:
                head_dim = key_states.shape[-1]
                cache_indices = indices[:, :, -1, :-block_size]
                cache_indices = cache_indices[..., None].expand(-1, -1, -1, head_dim)
                compressed_key = key_states.gather(dim=2, index=cache_indices)
                compressed_value = value_states.gather(dim=2, index=cache_indices)
                cache_kwargs = {
                    "cache_position": cache_position[: compressed_key.shape[-2]]
                    if cache_position is not None
                    else None
                }
                past_key_value.update(  # type: ignore
                    compressed_key,
                    compressed_value,
                    self.layer_idx,
                    cache_kwargs,
                )
                del compressed_key, compressed_value

            seq_len = query_states.shape[-2]
            pad = -seq_len % block_size
            if pad > 0:
                query_states = torch.nn.functional.pad(query_states, (0, 0, 0, pad))

            if self.config._attn_implementation == "flash_attention_2":
                attn_output, attn_weights = dynamic_attention_sinks_flash_attn_forward(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    block_size=block_size,
                    indices=indices[:, :, :-1],
                    origional_seq_len=seq_len,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                )
            else:
                attn_output, attn_weights = dynamic_attention_sinks_spda_forward(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    block_size=block_size,
                    indices=indices[:, :, :-1],
                    origional_seq_len=seq_len,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                )
        else:
            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_value.update(  # type: ignore
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


class LlamaAttentionOutputUnnormalized(LlamaAttention):
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

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(  # type: ignore
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_output_unnormalized_forward
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
        return attn_output, attn_weights


def update_llama_model_for_dynamic_attention_sinks(model):
    for i in range(len(model.model.layers)):
        attn_layer = model.model.layers[i].self_attn
        attn_layer.forward = LlamaAttentionDynamicAttentionSinks.forward.__get__(
            attn_layer,
            type(attn_layer),
        )

    return model


def update_llama_model_to_output_unnormalized_attentions(model):
    for i in range(len(model.model.layers)):
        attn_layer = model.model.layers[i].self_attn
        attn_layer.forward = LlamaAttentionOutputUnnormalized.forward.__get__(
            attn_layer,
            type(attn_layer),
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
