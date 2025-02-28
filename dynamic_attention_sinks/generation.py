from typing import Any

import torch
from torch import Tensor
from transformers import (  # type: ignore
    DynamicCache,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaAttention  # type: ignore
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention  # type: ignore

from .token_dropping_cache import TokenDroppingCache
from .sink_indices_util import get_sink_indices, update_indices


def generate_reduced_attentions(
    model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    generation_kwargs: dict[str, Any] = {},
) -> tuple[Tensor, Tensor]:
    if isinstance(model, LlamaForCausalLM):
        for layer in model.model.layers:
            assert isinstance(layer.self_attn, LlamaAttention)
    elif isinstance(model, Qwen2ForCausalLM):
        for layer in model.model.layers:
            assert isinstance(layer.self_attn, Qwen2Attention)
    else:
        raise NotImplementedError()

    input_len = input_ids.shape[1]
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

    for x in outputs.attentions:  # type: ignore
        for y in x:
            assert y.shape[2] == 1, y.shape

    reduced_attentions: list[Tensor] = [
        torch.cat([a[i][..., :input_len] for a in outputs.attentions], dim=2)  # type: ignore
        .square()
        .sum(dim=2)
        for i in range(model.config.num_hidden_layers)
    ]

    num_queries = model.config.num_attention_heads // model.config.num_key_value_heads
    if num_queries > 1:
        reduced_attentions = [
            a.view(
                input_ids.shape[0],
                model.config.num_key_value_heads,
                num_queries,
                input_len,
            ).mean(dim=2)
            for a in reduced_attentions
        ]

    return sequences, torch.stack(reduced_attentions)


def dynamic_attention_sinks_generate(
    model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    reduced_attentions: Tensor,
    block_size: int,
    k: int,
    generation_kwargs: dict[str, Any] = {},
) -> Tensor:
    if isinstance(model, LlamaForCausalLM):
        for layer in model.model.layers:
            assert isinstance(layer.self_attn, LlamaAttention)
    elif isinstance(model, Qwen2ForCausalLM):
        for layer in model.model.layers:
            assert isinstance(layer.self_attn, Qwen2Attention)
    else:
        raise NotImplementedError()

    assert input_ids.shape[0] == 1
    input_len = input_ids.shape[1]
    position_ids = torch.arange(input_ids, device=input_ids.device)[None]  # type: ignore

    k = min(k, input_len - block_size)

    sink_indices = get_sink_indices(
        reduced_attentions.sum(dim=[0, 2])[None, :, None],
        k=k,
        block_size=block_size,
    )[0, :, 0]

    past_key_values = TokenDroppingCache()
    cache_seq_indices: list[list[int]] = [[] for _ in range(input_ids.shape[0])]

    for block_idx in range((input_len + block_size - 1) // block_size):
        block_start = block_idx * block_size
        block_end = min((block_idx + 1) * block_size, input_len)
        block_input_ids = input_ids[:, block_start:block_end]
        block_position_ids = position_ids[:, block_start:block_end]

        with torch.no_grad():
            _ = model(
                input_ids=block_input_ids,
                position_ids=block_position_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )

        selected_indices: list[list[int]] = [[] for _ in range(input_ids.shape[0])]
        new_cache_seq_indices: list[list[int]] = [[] for _ in range(input_ids.shape[0])]
        for batch_idx in range(input_ids.shape[0]):
            indices = sink_indices[batch_idx][block_idx]

            cache_seq_indices[batch_idx] += list(range(block_start, block_end))
            for cache_idx, seq_idx in enumerate(cache_seq_indices[batch_idx]):
                if seq_idx in indices or seq_idx >= block_end - block_size:
                    selected_indices[batch_idx].append(cache_idx)
                    new_cache_seq_indices[batch_idx].append(seq_idx)

        past_key_values.token_select_indices(
            torch.tensor(selected_indices, device=input_ids.device)
        )

        cache_seq_indices = new_cache_seq_indices

    cache_size = min(block_size + k, input_len - 1)
    assert past_key_values.get_seq_length() == cache_size

    generated_ids: Tensor = model.generate(  # type: ignore
        input_ids=input_ids[:, -cache_size - 1 :],
        attention_mask=torch.ones_like(input_ids),
        use_cache=True,
        past_key_values=past_key_values,
        **generation_kwargs,
    )

    return torch.cat((input_ids[:, : -cache_size - 1], generated_ids), dim=1)


def dynamic_attention_sinks_generate_v2(
    model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    reduced_attentions: Tensor,
    block_size: int,
    k: int,
    generation_kwargs: dict[str, Any] = {},
) -> Tensor:
    if isinstance(model, LlamaForCausalLM):
        for layer in model.model.layers:
            assert isinstance(layer.self_attn, LlamaAttention)
    elif isinstance(model, Qwen2ForCausalLM):
        for layer in model.model.layers:
            assert isinstance(layer.self_attn, Qwen2Attention)
    else:
        raise NotImplementedError()

    assert input_ids.shape[0] == 1
    input_len = input_ids.shape[1]
    position_ids = torch.arange(input_ids, device=input_ids.device)[None]  # type: ignore

    k = min(k, input_len - block_size)

    sink_indices = get_sink_indices(reduced_attentions, k=k, block_size=block_size)

    past_key_values = TokenDroppingCache()

    cache_seq_indices = torch.empty(
        model.config.num_hidden_layers,
        input_ids.shape[0],
        model.config.num_key_value_heads,
        0,
        dtype=torch.int64,
    )

    for block_idx in range((input_len + block_size - 1) // block_size):
        block_start = block_idx * block_size
        block_end = min((block_idx + 1) * block_size, input_len)
        block_input_ids = input_ids[:, block_start:block_end]
        block_position_ids = position_ids[:, block_start:block_end]

        with torch.no_grad():
            _ = model(
                input_ids=block_input_ids,
                position_ids=block_position_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )

        selected_indices, cache_seq_indices = update_indices(
            sink_indices=sink_indices,
            cache_seq_indices=cache_seq_indices,
            block_idx=block_idx,
            block_size=block_size,
            k=k,
            input_len=input_len,
        )

        for layer_idx in range(model.config.num_hidden_layers):
            past_key_values.token_select_indices(
                selected_indices[layer_idx],
                layer_idx=layer_idx,
            )

    cache_size = min(block_size + k, input_len - 1)
    assert past_key_values.get_seq_length() == cache_size

    generated_ids: Tensor = model.generate(  # type: ignore
        input_ids=input_ids[:, -cache_size - 1 :],
        attention_mask=torch.ones_like(input_ids),
        use_cache=True,
        past_key_values=past_key_values,
        **generation_kwargs,
    )

    return torch.cat((input_ids[:, : -cache_size - 1], generated_ids), dim=1)
