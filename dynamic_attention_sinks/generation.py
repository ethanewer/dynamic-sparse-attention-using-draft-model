from typing import Any, Literal

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
from .indices_util import get_cache_update_indices


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

    ks = [max_k - (max_k - min_k) * i // (m - 1) for i in range(m)]
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
    if isinstance(model, LlamaForCausalLM):
        for layer in model.model.layers:
            assert isinstance(layer.self_attn, LlamaAttention)
    elif isinstance(model, Qwen2ForCausalLM):
        for layer in model.model.layers:
            assert isinstance(layer.self_attn, Qwen2Attention)
    else:
        raise NotImplementedError()

    assert input_ids.shape[0] == 1
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

        if isinstance(cache_update_indices, list):
            for layer_idx in range(model.config.num_hidden_layers):
                past_key_values.token_select_indices(
                    cache_update_indices[layer_idx][block_idx],
                    layer_idx=layer_idx,
                )
        else:
            past_key_values.token_select_indices(cache_update_indices[block_idx])

    cache_size = min(block_size + k, prefill_input_len)
    assert past_key_values.get_seq_length() == cache_size, (
        past_key_values.get_seq_length(),
        prefill_input_len,
        block_size + k,
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
    if isinstance(model, LlamaForCausalLM):
        for layer in model.model.layers:
            assert isinstance(layer.self_attn, LlamaAttention)
    elif isinstance(model, Qwen2ForCausalLM):
        for layer in model.model.layers:
            assert isinstance(layer.self_attn, Qwen2Attention)
    else:
        raise NotImplementedError()

    assert input_ids.shape[0] == 1
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

    cache_size = min(block_size + k, prefill_input_len)
    assert past_key_values.get_seq_length() == cache_size

    generated_ids: Tensor = model.generate(  # type: ignore
        input_ids=input_ids[:, -cache_size - 1 :],
        attention_mask=torch.ones_like(input_ids),
        use_cache=True,
        past_key_values=past_key_values,
        **generation_kwargs,
    )

    return torch.cat((input_ids[:, : -cache_size - 1], generated_ids), dim=1)
