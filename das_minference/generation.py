from typing import Any

import torch
from torch import Tensor
from transformers import (  # type: ignore
    DynamicCache,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
)

from .llama_util import update_llama_model_for_das_minference
from .qwen2_util import update_qwen2_model_for_das_minference


def generate_reduced_attentions(
    model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    generation_kwargs: dict[str, Any] = {},
) -> tuple[Tensor, Tensor]:
    past_key_values = DynamicCache()

    with torch.no_grad():
        _ = model(
            input_ids=input_ids[:, :-1],
            past_key_values=past_key_values,
            use_cache=True,
        )

    outputs = model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        output_attentions=True,
        use_cache=True,
        past_key_values=past_key_values,
        return_dict_in_generate=True,
        **generation_kwargs,
    )

    sequences: Tensor = outputs.sequences  # type: ignore
    attentions: tuple[tuple[Tensor, ...], ...] = outputs.attentions  # type: ignore

    reduced_attentions = [
        torch.cat([a[i][..., : input_ids.shape[1]] for a in attentions], dim=2).mean(
            dim=2
        )
        for i in range(model.config.num_hidden_layers)
    ]

    num_queries = model.config.num_attention_heads // model.config.num_key_value_heads
    if num_queries > 1:
        reduced_attentions = [
            a.view(
                a.shape[0],
                model.config.num_key_value_heads,
                num_queries,
                -1,
            ).mean(dim=2)
            for a in reduced_attentions
        ]

    reduced_attentions = torch.stack(reduced_attentions)

    return sequences, reduced_attentions


# def das_minference_generate(
#     model: LlamaForCausalLM | Qwen2ForCausalLM,
#     input_ids: Tensor,
#     reduced_attentions: Tensor,
#     window_size: int,
#     max_capacity_prompt: int,
#     generation_kwargs: dict[str, Any] = {},
# ) -> Tensor:
#     if isinstance(model, LlamaForCausalLM):
#         update_llama_model_for_das_minference(model)
#     elif isinstance(model, Qwen2ForCausalLM):
#         update_qwen2_model_for_das_minference(model)
#     else:
#         raise NotImplementedError()

#     prefill_input_len = input_ids.shape[1] - 1

#     assert window_size < max_capacity_prompt and window_size % 64 == 0
#     k = min(max_capacity_prompt, prefill_input_len) - window_size

#     if k <= 0:
#         return model.generate(  # type: ignore
#             input_ids=input_ids,
#             attention_mask=torch.ones_like(input_ids),
#             use_cache=True,
#             **generation_kwargs,
#         )

#     v_idx = reduced_attentions[..., :prefill_input_len].topk(k, dim=-1).indices.int()
#     s_idx = torch.arange(
#         window_size,
#         -1,
#         -64,
#         dtype=v_idx.dtype,
#         device=v_idx.device,
#     )[None, None, None].expand(*v_idx.shape[:-1], -1)

#     past_key_values = DynamicCache()

#     with torch.no_grad():
#         _ = model(
#             input_ids=input_ids[:, :-1],
#             use_cache=True,
#             past_key_values=past_key_values,
#             v_idx=v_idx,
#             s_idx=s_idx,
#             window_size=window_size,
#         )

#     cache_size = past_key_values.get_seq_length()
#     for layer_idx in range(model.config.num_hidden_layers):
#         assert past_key_values.get_seq_length(layer_idx) == window_size + k

#     generated_ids: Tensor = model.generate(  # type: ignore
#         input_ids=input_ids[:, -cache_size - 1 :],
#         attention_mask=torch.ones_like(input_ids),
#         use_cache=True,
#         past_key_values=past_key_values,
#         **generation_kwargs,
#     )

#     sequences = torch.cat((input_ids[:, : -cache_size - 1], generated_ids), dim=1)

#     num_generated_tokens = sequences.shape[1] - input_ids.shape[1]
#     for layer_idx in range(model.config.num_hidden_layers):
#         assert (
#             past_key_values.get_seq_length(layer_idx)
#             == window_size + k + num_generated_tokens
#         )

#     return sequences


def das_minference_generate(
    model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    reduced_attentions: Tensor,
    window_size: int,
    max_capacity_prompt: int,
    generation_kwargs: dict[str, Any] = {},
) -> Tensor:
    if isinstance(model, LlamaForCausalLM):
        update_llama_model_for_das_minference(model)
    elif isinstance(model, Qwen2ForCausalLM):
        update_qwen2_model_for_das_minference(model)
    else:
        raise NotImplementedError()

    assert window_size < max_capacity_prompt and window_size % 64 == 0
    k = min(max_capacity_prompt, input_ids.shape[1]) - window_size

    if k <= 0:
        return model.generate(  # type: ignore
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=True,
            **generation_kwargs,
        )

    v_idx = reduced_attentions[..., : input_ids.shape[1]].topk(k, dim=-1).indices.int()
    s_idx = torch.arange(
        window_size,
        -1,
        -64,
        dtype=v_idx.dtype,
        device=v_idx.device,
    )[None, None, None].expand(*v_idx.shape[:-1], -1)

    past_key_values = DynamicCache()

    output_ids: Tensor = model.generate(  # type: ignore
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=True,
        past_key_values=past_key_values,
        kwargs=dict(
            v_idx=v_idx,
            s_idx=s_idx,
            window_size=window_size,
        ),
        **generation_kwargs,
    )

    expected_cache_size = window_size + k + output_ids.shape[1] - input_ids.shape[1] - 1
    for layer_idx in range(model.config.num_hidden_layers):
        assert past_key_values.get_seq_length(layer_idx) == expected_cache_size, (
            window_size + k,
            output_ids.shape[1] - input_ids.shape[1],
            expected_cache_size,
            past_key_values.get_seq_length(layer_idx),
            layer_idx,
        )

    return output_ids
