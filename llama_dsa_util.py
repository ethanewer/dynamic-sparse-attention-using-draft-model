from typing import Callable, Optional

import torch
from transformers import Cache  # type: ignore
from transformers.models.llama.modeling_llama import (  # type: ignore
    ALL_ATTENTION_FUNCTIONS,
    LlamaAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
    logger,
)


class LlamaAttentionDSA(LlamaAttention):
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

        ################################ NEW CODE ################################
        # if (
        #     query_states.shape[0] == 1
        #     and query_states.shape[2] == 1
        #     and attention_mask is not None
        #     and not output_attentions
        # ):
        #     bool_mask = attention_mask[0, 0, 0] >= 0
        #     if not bool_mask.all():
        #         key_states = key_states[:, :, bool_mask]
        #         value_states = value_states[:, :, bool_mask]
        #         attention_mask = None
        if (
            query_states.shape[0] == 1
            and query_states.shape[2] == 1
            and attention_mask is not None
            and not kwargs.get("output_attentions", False)
        ):
            idxs = (attention_mask[0, 0, 0] >= 0).nonzero(as_tuple=True)[0]
            if len(idxs) < key_states.shape[2]:
                key_states = key_states[:, :, idxs]
                value_states = value_states[:, :, idxs]

            attention_mask = None
        ##########################################################################

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


def update_llama_model_for_dsa(model):
    for i in range(len(model.model.layers)):
        attn_layer = model.model.layers[i].self_attn
        attn_layer.forward = LlamaAttentionDSA.forward.__get__(
            attn_layer,
            type(attn_layer),
        )

    return model
