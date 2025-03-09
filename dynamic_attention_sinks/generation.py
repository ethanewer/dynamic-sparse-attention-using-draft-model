from typing import Any, Literal

import torch
from torch import Tensor
from transformers import (  # type: ignore
    DynamicCache,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
)

from .indices_util import get_cache_update_indices, get_indices_and_attention_mask
from .llama_util import update_llama_model_for_dynamic_attention_sinks
from .qwen2_util import update_qwen2_model_for_dynamic_attention_sinks
from .token_dropping_cache import TokenDroppingCache


def reduce_attentions(
    attentions: tuple[tuple[Tensor, ...], ...],
    reduction: Literal["mean", "squared_sum", "max"],
    batch_size: int,
    input_len: int,
    config: Any,
) -> Tensor:
    if reduction == "mean":
        reduced_attentions = [
            torch.cat([a[i][..., :input_len] for a in attentions], dim=2).mean(dim=2)
            for i in range(config.num_hidden_layers)
        ]
    elif reduction == "squared_sum":
        reduced_attentions = [
            torch.cat([a[i][..., :input_len] for a in attentions], dim=2)
            .square()
            .sum(dim=2)
            for i in range(config.num_hidden_layers)
        ]
    elif reduction == "max":
        reduced_attentions = [
            torch.cat([a[i][..., :input_len] for a in attentions], dim=2)
            .max(dim=2)
            .values
            for i in range(config.num_hidden_layers)
        ]
    else:
        raise NotImplementedError

    num_queries = config.num_attention_heads // config.num_key_value_heads
    if num_queries > 1:
        reduced_attentions = [
            a.view(
                batch_size,
                config.num_key_value_heads,
                num_queries,
                input_len,
            ).mean(dim=2)
            for a in reduced_attentions
        ]

    return torch.stack(reduced_attentions)


def generate_reduced_attentions(
    model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    reduction: Literal["mean", "squared_sum", "max"],
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

    for x in outputs.attentions:  # type: ignore
        for y in x:
            assert y.shape[2] == 1, y.shape

    attentions: tuple[tuple[Tensor, ...], ...] = outputs.attentions  # type: ignore

    reduced_attentions = reduce_attentions(
        attentions=attentions,
        reduction=reduction,
        batch_size=input_ids.shape[0],
        input_len=input_ids.shape[1],
        config=model.config,
    )

    return sequences, reduced_attentions


def get_pyramid_ks(
    k: int,
    m: int,
    beta: int,
    prefill_input_len: int,
    block_size: int,
) -> list[int]:
    min_k = k // beta
    max_k = 2 * k - min_k

    if max_k >= prefill_input_len - block_size:
        max_k = prefill_input_len - block_size
        min_k = 2 * k * m - max_k

    ks = [int(max_k - (max_k - min_k) * i / (m - 1)) for i in range(m)]
    return ks


def dynamic_attention_sinks_generate(
    model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    reduced_attentions: Tensor,
    block_size: int,
    k: int,
    layer_aggregation: Literal["mean", "last"] = "mean",
    pyramid: bool = False,
    beta: int = 8,
    generation_kwargs: dict[str, Any] = {},
) -> Tensor:
    prefill_input_len = input_ids.shape[1] - 1
    position_ids = torch.arange(prefill_input_len, device=input_ids.device)[None]  # type: ignore

    k = min(k, prefill_input_len - block_size)

    if layer_aggregation == "mean":
        aggregate_attentions = reduced_attentions.mean(dim=(0, 2))
    elif layer_aggregation == "last":
        aggregate_attentions = reduced_attentions[-1].mean(dim=1)
    else:
        raise NotImplementedError

    if pyramid:
        layer_ks = get_pyramid_ks(
            k=k,
            m=model.config.num_hidden_layers,
            beta=beta,
            prefill_input_len=prefill_input_len,
            block_size=block_size,
        )
        cache_update_indices = [
            get_cache_update_indices(
                reduced_attentions=aggregate_attentions[None, :, None, :-1],
                k=layer_k,
                block_size=block_size,
                reduce_heads=True,
                device=input_ids.device,  # type: ignore
            )
            for layer_k in layer_ks
        ]
    else:
        cache_update_indices = get_cache_update_indices(
            reduced_attentions=aggregate_attentions[None, :, None, :-1],
            k=k,
            block_size=block_size,
            reduce_heads=True,
            device=input_ids.device,  # type: ignore
        )

    past_key_values = TokenDroppingCache()

    for block_idx in range((prefill_input_len + block_size - 1) // block_size):
        block_start = block_idx * block_size
        block_end = min((block_idx + 1) * block_size, prefill_input_len)
        block_input_ids = input_ids[:, block_start:block_end]
        block_position_ids = position_ids[:, block_start:block_end]

        with torch.no_grad():
            _ = model(
                input_ids=block_input_ids,
                position_ids=block_position_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )

        if pyramid:
            for layer_idx in range(model.config.num_hidden_layers):
                past_key_values.token_select_indices(
                    cache_update_indices[layer_idx][block_idx],
                    layer_idx=layer_idx,
                )
        else:
            past_key_values.token_select_indices(cache_update_indices[block_idx])  # type: ignore

    cache_size = past_key_values.get_seq_length()
    full_cache_size = sum(
        past_key_values.get_seq_length(layer_idx)
        for layer_idx in range(model.config.num_hidden_layers)
    )
    max_full_cache_size = (
        min(block_size + k, prefill_input_len) * model.config.num_hidden_layers
    )
    assert full_cache_size <= max_full_cache_size, (
        full_cache_size,
        max_full_cache_size,
    )
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
    prefill_input_len = input_ids.shape[1] - 1
    position_ids = torch.arange(prefill_input_len, device=input_ids.device)[None]  # type: ignore

    k = min(k, prefill_input_len - block_size)

    cache_update_indices = get_cache_update_indices(
        reduced_attentions[..., :-1],
        k=k,
        block_size=block_size,
        reduce_heads=False,
        device=input_ids.device,  # type: ignore
    )

    past_key_values = TokenDroppingCache()

    for block_idx in range((prefill_input_len + block_size - 1) // block_size):
        block_start = block_idx * block_size
        block_end = min((block_idx + 1) * block_size, prefill_input_len)
        block_input_ids = input_ids[:, block_start:block_end]
        block_position_ids = position_ids[:, block_start:block_end]

        with torch.no_grad():
            _ = model(
                input_ids=block_input_ids,
                position_ids=block_position_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )

        for layer_idx in range(model.config.num_hidden_layers):
            past_key_values.token_select_indices(
                cache_update_indices[block_idx][layer_idx],
                layer_idx=layer_idx,
            )

    cache_size = past_key_values.get_seq_length()
    assert cache_size == min(block_size + k, prefill_input_len)

    generated_ids: Tensor = model.generate(  # type: ignore
        input_ids=input_ids[:, -cache_size - 1 :],
        attention_mask=torch.ones_like(input_ids),
        use_cache=True,
        past_key_values=past_key_values,
        **generation_kwargs,
    )

    return torch.cat((input_ids[:, : -cache_size - 1], generated_ids), dim=1)


def dynamic_attention_sinks_generate_v3(
    model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    reduced_attentions: Tensor,
    block_size: int,
    k: int,
    generation_kwargs: dict[str, Any] = {},
) -> Tensor:
    if isinstance(model, LlamaForCausalLM):
        update_llama_model_for_dynamic_attention_sinks(model)
    elif isinstance(model, Qwen2ForCausalLM):
        update_qwen2_model_for_dynamic_attention_sinks(model)
    else:
        raise NotImplementedError()

    prefill_input_len = input_ids.shape[1] - 1

    k = min(k, prefill_input_len - block_size)

    indices, attention_mask = get_indices_and_attention_mask(
        input_ids=input_ids[:, :-1],
        reduced_attentions=reduced_attentions[..., :-1],
        k=k,
        block_size=block_size,
        dtype=model.dtype,
    )

    past_key_values = DynamicCache()

    with torch.no_grad():
        _ = model(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
            dynamic_attention_sinks_block_size=block_size,
            dynamic_attention_sinks_indices=indices,
        )

    past_key_values.crop(prefill_input_len)

    cache_size = past_key_values.get_seq_length()
    assert cache_size == min(block_size + k, prefill_input_len)

    generated_ids: Tensor = model.generate(  # type: ignore
        input_ids=input_ids[:, -cache_size - 1 :],
        attention_mask=torch.ones_like(input_ids),
        use_cache=True,
        past_key_values=past_key_values,
        **generation_kwargs,
    )

    return torch.cat((input_ids[:, : -cache_size - 1], generated_ids), dim=1)
