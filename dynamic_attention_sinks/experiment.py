from typing import Callable

import torch
from torch import Tensor
from transformers import LlamaForCausalLM, Qwen2ForCausalLM  # type: ignore
from transformers.models.llama.modeling_llama import LlamaAttention  # type: ignore
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention  # type: ignore

from .token_dropping_cache import TokenDroppingCache
from .indices_util import get_cache_update_indices


def streaming_llm_experiment(
    model: Callable,
    input_ids: Tensor,
    generated_ids: Tensor,
    window_size: int,
    n_sinks: int,
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
    position_ids = torch.arange(input_len, device=input_ids.device)[None]

    mask = torch.ones(input_len, input_len).tril()
    mask[window_size:, :-window_size] -= torch.ones(
        input_len - window_size, input_len - window_size
    ).tril()

    n_sinks = min(n_sinks, input_len)
    for i in range(n_sinks):
        mask[:, i] = 1

    mask.tril_()

    mask_4d = -3.4028e38 * (1 - mask[None, None, :, :]).to(
        input_ids.device,
        model.dtype,  # type: ignore
    )

    past_key_values = TokenDroppingCache()
    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=mask_4d,
            use_cache=True,
            past_key_values=past_key_values,
        )

    logits = []
    for i in range(input_len + 1, generated_ids.shape[1]):
        input_ids = generated_ids[:, i - 1 : i]
        position_ids = torch.tensor([[i - 1]], device=input_ids.device)

        if past_key_values.get_seq_length() >= window_size + n_sinks:
            sink_indices = list(range(n_sinks))
            window_indices = list(
                range(
                    past_key_values.get_seq_length() - window_size + 1,
                    past_key_values.get_seq_length(),
                )
            )
            past_key_values.token_select_indices(
                torch.tensor(sink_indices + window_indices, device=input_ids.device)
            )

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )

        logits.append(outputs.logits[0, -1:])

    return torch.cat(logits).float().cpu()


def dynamic_attention_sinks_experiment(
    model: Callable,
    input_ids: Tensor,
    generated_ids: Tensor,
    reduced_attentions: Tensor,
    block_size: int,
    k: int,
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
    position_ids = torch.arange(input_len, device=input_ids.device)[None]

    k = min(k, input_len - block_size)

    cache_update_indices = get_cache_update_indices(
        reduced_attentions.sum(dim=[0, 2])[None, :, None],
        k=k,
        block_size=block_size,
        reduce_heads=True,
        device=input_ids.device,  # type: ignore
    )

    past_key_values = TokenDroppingCache()

    for block_idx in range((input_len + block_size - 1) // block_size):
        block_start = block_idx * block_size
        block_end = min((block_idx + 1) * block_size, input_len)
        block_input_ids = input_ids[:, block_start:block_end]
        block_position_ids = position_ids[:, block_start:block_end]

        with torch.no_grad():
            outputs = model(
                input_ids=block_input_ids,
                position_ids=block_position_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )

        past_key_values.token_select_indices(cache_update_indices[block_idx])

    assert past_key_values.get_seq_length() == block_size + k, (
        past_key_values.get_seq_length(),
        block_size,
        k,
    )

    logits = []
    for i in range(input_len + 1, generated_ids.shape[1]):
        input_ids = generated_ids[:, i - 1 : i]
        position_ids = torch.tensor([[i - 1]], device=input_ids.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )

        logits.append(outputs.logits[0, -1:])

    return torch.cat(logits).float().cpu()


def dynamic_attention_sinks_v2_experiment(
    model: Callable,
    input_ids: Tensor,
    generated_ids: Tensor,
    reduced_attentions: Tensor,
    block_size: int,
    k: int,
) -> Tensor:
    if isinstance(model, LlamaForCausalLM):
        for layer in model.model.layers:
            assert isinstance(layer.self_attn, LlamaAttention)
    elif isinstance(model, Qwen2ForCausalLM):
        for layer in model.model.layers:
            assert isinstance(layer.self_attn, Qwen2Attention)
    else:
        raise NotImplementedError()

    input_len = input_ids.shape[1]

    position_ids = torch.arange(input_len, device=input_ids.device)[None]

    k = min(k, input_len - block_size)

    cache_update_indices = get_cache_update_indices(
        reduced_attentions,
        k=k,
        block_size=block_size,
        reduce_heads=False,
        device=input_ids.device,  # type: ignore
    )

    past_key_values = TokenDroppingCache()

    for block_idx in range((input_len + block_size - 1) // block_size):
        block_start = block_idx * block_size
        block_end = min((block_idx + 1) * block_size, input_len)
        block_input_ids = input_ids[:, block_start:block_end]
        block_position_ids = position_ids[:, block_start:block_end]

        with torch.no_grad():
            outputs = model(
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

    assert past_key_values.get_seq_length() == block_size + k

    logits = []
    for i in range(input_len + 1, generated_ids.shape[1]):
        input_ids = generated_ids[:, i - 1 : i]
        position_ids = torch.tensor([[i - 1]], device=input_ids.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )

        logits.append(outputs.logits[0, -1:])

    return torch.cat(logits).float().cpu()
