from typing import Callable, Optional

import torch
from transformers import Cache  # type: ignore
from transformers.models.qwen2.modeling_qwen2 import (  # type: ignore
    ALL_ATTENTION_FUNCTIONS,
    Qwen2Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
    logger,
)


class Qwen2AttentionDSA(Qwen2Attention):
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

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        ################################ NEW CODE ################################
        if (
            "dsa_k" in kwargs
            and query_states.shape[2] == 1
            and attention_mask is not None
            and not kwargs.get("output_attentions", False)
        ):
            k = kwargs["dsa_k"]
            if k < key_states.shape[2]:
                indices = attention_mask[:, 0, 0].topk(k, dim=-1).indices
                key_value_indices = indices[:, None, :, None].expand(
                    -1,
                    key_states.shape[1],
                    -1,
                    key_states.shape[3],
                )
                attention_mask_indices = indices[:, None, None, :].expand(
                    -1,
                    attention_mask.shape[1],
                    attention_mask.shape[2],
                    -1,
                )
                key_states = key_states.gather(dim=2, index=key_value_indices)
                value_states = value_states.gather(dim=2, index=key_value_indices)
                attention_mask = attention_mask.gather(
                    dim=3,
                    index=attention_mask_indices,
                )
        ##########################################################################

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            # sliding_window = self.config.sliding_window
            logger.warning_once(  # type: ignore
                "Dynamic sparse attention is not compatible with sliding window."
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
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights  # type: ignore


def update_qwen2_model_for_dsa(model):
    for i in range(len(model.model.layers)):
        attn_layer = model.model.layers[i].self_attn
        attn_layer.forward = Qwen2AttentionDSA.forward.__get__(
            attn_layer,
            type(attn_layer),
        )

    return model
