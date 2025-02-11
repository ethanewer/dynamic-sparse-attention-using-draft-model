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


# class Qwen2FlashAttention2DSA(Qwen2FlashAttention2):
#     def forward(
#         self,
#         hidden_states: Tensor,
#         attention_mask: Optional[Tensor] = None,
#         position_ids: Optional[LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         cache_position: Optional[LongTensor] = None,
#         position_embeddings: Optional[
#             tuple[Tensor, Tensor]
#         ] = None,  # will become mandatory in v4.46
#     ):
#         bsz, q_len, _ = hidden_states.size()

#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)

#         query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
#         key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
#         value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

#         if position_embeddings is None:
#             cos, sin = self.rotary_emb(value_states, position_ids)
#         else:
#             cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(
#             query_states, key_states, cos, sin
#         )

#         if past_key_value is not None:
#             cache_kwargs = {
#                 "sin": sin,
#                 "cos": cos,
#                 "cache_position": cache_position,
#             }  # Specific to RoPE models
#             key_states, value_states = past_key_value.update(
#                 key_states,
#                 value_states,
#                 self.layer_idx,  # type: ignore
#                 cache_kwargs,
#             )

#         # repeat k/v heads if n_kv_heads < n_heads
#         key_states = repeat_kv(key_states, self.num_key_value_groups)
#         value_states = repeat_kv(value_states, self.num_key_value_groups)
#         dropout_rate = 0.0 if not self.training else self.attention_dropout

#         # In PEFT, usually we cast the layer norms in float32 for training stability reasons
#         # therefore the input hidden states gets silently casted in float32. Hence, we need
#         # cast them back in float16 just to be sure everything works as expected.
#         input_dtype = query_states.dtype
#         if input_dtype == torch.float32:
#             if torch.is_autocast_enabled():
#                 target_dtype = torch.get_autocast_gpu_dtype()
#             # Handle the case where the model is quantized
#             elif hasattr(self.config, "_pre_quantization_dtype"):
#                 target_dtype = self.config._pre_quantization_dtype
#             else:
#                 target_dtype = self.q_proj.weight.dtype

#             query_states = query_states.to(target_dtype)
#             key_states = key_states.to(target_dtype)
#             value_states = value_states.to(target_dtype)

#         # Reashape to the expected shape for Flash Attention
#         query_states = query_states.transpose(1, 2)
#         key_states = key_states.transpose(1, 2)
#         value_states = value_states.transpose(1, 2)

#         if (
#             self.config.use_sliding_window
#             and getattr(self.config, "sliding_window", None) is not None
#             and self.layer_idx >= self.config.max_window_layers  # type: ignore
#         ):
#             sliding_window = self.config.sliding_window
#         else:
#             sliding_window = None

#         ################################ NEW CODE ################################
#         # if (
#         #     query_states.shape[0] == 1
#         #     and query_states.shape[2] == 1
#         #     and attention_mask is not None
#         #     and not output_attentions
#         # ):
#         #     bool_mask = attention_mask[0, 0, 0] >= 0
#         #     if not bool_mask.all():
#         #         key_states = key_states[:, :, bool_mask]
#         #         value_states = value_states[:, :, bool_mask]
#         #         attention_mask = None
#         if (
#             query_states.shape[0] == 1
#             and query_states.shape[2] == 1
#             and attention_mask is not None
#             and not output_attentions
#         ):
#             idxs = (attention_mask[0, 0, 0] >= 0).nonzero(as_tuple=True)[0]
#             if len(idxs) < key_states.shape[2]:
#                 key_states = key_states[:, :, idxs]
#                 value_states = value_states[:, :, idxs]

#             attention_mask = None
#         ##########################################################################

#         attn_output = _flash_attention_forward(  # type: ignore
#             query_states,
#             key_states,
#             value_states,
#             attention_mask,  # type: ignore
#             q_len,
#             position_ids=position_ids,
#             dropout=dropout_rate,
#             sliding_window=sliding_window,
#             is_causal=self.is_causal,
#             use_top_left_mask=self._flash_attn_uses_top_left_mask,
#         )

#         attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
#         attn_output = self.o_proj(attn_output)

#         if not output_attentions:
#             attn_weights = None

#         return attn_output, attn_weights, past_key_value  # type: ignore


# class Qwen2SdpaAttentionDSA(Qwen2SdpaAttention):
#     def forward(
#         self,
#         hidden_states: Tensor,
#         attention_mask: Optional[Tensor] = None,
#         position_ids: Optional[LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         cache_position: Optional[LongTensor] = None,
#         position_embeddings: Optional[
#             tuple[Tensor, Tensor]
#         ] = None,  # will become mandatory in v4.46
#     ) -> tuple[Tensor, Optional[Tensor], Optional[tuple[Tensor]]]:
#         bsz, q_len, _ = hidden_states.size()

#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)

#         query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
#         key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
#         value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

#         if position_embeddings is None:
#             cos, sin = self.rotary_emb(value_states, position_ids)
#         else:
#             cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(
#             query_states, key_states, cos, sin
#         )

#         if past_key_value is not None:
#             cache_kwargs = {
#                 "sin": sin,
#                 "cos": cos,
#                 "cache_position": cache_position,
#             }  # Specific to RoPE models
#             key_states, value_states = past_key_value.update(
#                 key_states,
#                 value_states,
#                 self.layer_idx,  # type: ignore
#                 cache_kwargs,
#             )

#         key_states = repeat_kv(key_states, self.num_key_value_groups)
#         value_states = repeat_kv(value_states, self.num_key_value_groups)

#         ################################ NEW CODE ################################
#         # if (
#         #     query_states.shape[0] == 1
#         #     and query_states.shape[2] == 1
#         #     and attention_mask is not None
#         #     and not output_attentions
#         # ):
#         #     bool_mask = attention_mask[0, 0, 0] >= 0
#         #     if not bool_mask.all():
#         #         key_states = key_states[:, :, bool_mask]
#         #         value_states = value_states[:, :, bool_mask]
#         #         attention_mask = None
#         if (
#             query_states.shape[0] == 1
#             and query_states.shape[2] == 1
#             and attention_mask is not None
#             and not output_attentions
#         ):
#             idxs = (attention_mask[0, 0, 0] >= 0).nonzero(as_tuple=True)[0]
#             if len(idxs) < key_states.shape[2]:
#                 key_states = key_states[:, :, idxs]
#                 value_states = value_states[:, :, idxs]

#             attention_mask = None
#         ##########################################################################

#         if output_attentions:
#             attn_weights = torch.matmul(
#                 query_states, key_states.transpose(2, 3)
#             ) / math.sqrt(self.head_dim)
#             if attention_mask is not None:  # no matter the length, we just slice it
#                 causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
#                 attn_weights = attn_weights + causal_mask

#             # upcast attention to fp32
#             attn_weights = nn.functional.softmax(
#                 attn_weights, dim=-1, dtype=torch.float32
#             ).to(query_states.dtype)
#             attn_weights = nn.functional.dropout(
#                 attn_weights, p=self.attention_dropout, training=self.training
#             )
#             attn_output = torch.matmul(attn_weights, value_states)

#             if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
#                 raise ValueError(
#                     f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
#                     f" {attn_output.size()}"
#                 )

#             attn_output = attn_output.transpose(1, 2).contiguous()
#             attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

#             attn_output = self.o_proj(attn_output)

#             if not output_attentions:
#                 attn_weights = None  # type: ignore

#             return attn_output, attn_weights, past_key_value  # type: ignore
#         else:
#             causal_mask = attention_mask  # type: ignore
#             if attention_mask is not None:  # no matter the length, we just slice it
#                 causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

#             # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
#             # Reference: https://github.com/pytorch/pytorch/issues/112577.
#             if query_states.device.type == "cuda" and attention_mask is not None:
#                 query_states = query_states.contiguous()
#                 key_states = key_states.contiguous()
#                 value_states = value_states.contiguous()

#             # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
#             # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
#             # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
#             is_causal = True if causal_mask is None and q_len > 1 else False

#             attn_output = torch.nn.functional.scaled_dot_product_attention(
#                 query_states,
#                 key_states,
#                 value_states,
#                 attn_mask=causal_mask,
#                 dropout_p=self.attention_dropout if self.training else 0.0,
#                 is_causal=is_causal,
#             )

#             attn_output = attn_output.transpose(1, 2).contiguous()
#             attn_output = attn_output.view(bsz, q_len, self.hidden_size)

#             attn_output = self.o_proj(attn_output)

#             return attn_output, None, past_key_value  # type: ignore

# def update_qwen2_model_for_dsa(model):
#     for i in range(len(model.model.layers)):
#         attn_layer = model.model.layers[i].self_attn

#         if isinstance(attn_layer, Qwen2FlashAttention2):
#             attn_layer.forward = Qwen2FlashAttention2DSA.forward.__get__(
#                 attn_layer,
#                 type(attn_layer),
#             )
#         elif isinstance(attn_layer, Qwen2SdpaAttention):
#             attn_layer.forward = Qwen2SdpaAttentionDSA.forward.__get__(
#                 attn_layer,
#                 type(attn_layer),
#             )

#     return model
