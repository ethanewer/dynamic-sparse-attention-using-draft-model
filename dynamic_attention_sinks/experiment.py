from typing import Callable

import torch
from torch import Tensor
from transformers import LlamaForCausalLM, Qwen2ForCausalLM  # type: ignore
from transformers.models.llama.modeling_llama import LlamaAttention  # type: ignore
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention  # type: ignore

from .token_dropping_cache import TokenDroppingCache


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
            past_key_values.token_select_indices(sink_indices + window_indices)

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
    sink_indices = reduced_attentions[: input_len - block_size].topk(k).indices.tolist()
    cache_seq_indices = []
    past_key_values = TokenDroppingCache()

    for i in range(0, input_ids.shape[1], block_size):
        j = min(i + block_size, input_ids.shape[1])
        block_input_ids = input_ids[:, i:j]
        block_position_ids = position_ids[:, i:j]

        with torch.no_grad():
            outputs = model(
                input_ids=block_input_ids,
                position_ids=block_position_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )

        cache_seq_indices += list(range(i, j))
        selected_indices = []
        new_cache_seq_indices = []
        for cache_idx, seq_idx in enumerate(cache_seq_indices):
            if seq_idx in sink_indices or seq_idx >= j - block_size:
                selected_indices.append(cache_idx)
                new_cache_seq_indices.append(seq_idx)

        past_key_values.token_select_indices(
            torch.tensor(selected_indices, device=input_ids.device)
        )
        cache_seq_indices = new_cache_seq_indices

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


def dynamic_attention_sinks_v2_experiment(
    model: Callable,
    input_ids: Tensor,
    generated_ids: Tensor,
    reduced_attention_matrix: Tensor,
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
    assert input_ids.shape[1] == reduced_attention_matrix.shape[1]
    input_len = input_ids.shape[1]
    position_ids = torch.arange(input_len, device=input_ids.device)[None]

    k = min(k, input_len - block_size)
    sink_indices: list[list[int]] = [[]]
    for i in range(block_size, input_len, block_size):
        indices = sink_indices[-1] + list(
            range(i - block_size, min(i, input_len - block_size))
        )
        a = reduced_attention_matrix[i + block_size :, indices].square().sum(dim=0)
        sink_indices.append([indices[i] for i in a.topk(k).indices])

    cache_seq_indices = []
    past_key_values = TokenDroppingCache()
    print(reduced_attention_matrix.shape)

    for i, indices in enumerate(sink_indices):
        block_start = i * block_size
        block_end = min((i + 1) * block_size, input_ids.shape[1])
        block_input_ids = input_ids[:, block_start:block_end]
        block_position_ids = position_ids[:, block_start:block_end]

        with torch.no_grad():
            outputs = model(
                input_ids=block_input_ids,
                position_ids=block_position_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )

        cache_seq_indices += list(range(block_start, block_end))
        selected_indices = []
        new_cache_seq_indices = []
        for cache_idx, seq_idx in enumerate(cache_seq_indices):
            if seq_idx in indices or seq_idx >= block_end - block_size:
                selected_indices.append(cache_idx)
                new_cache_seq_indices.append(seq_idx)

        past_key_values.token_select_indices(selected_indices)
        cache_seq_indices = new_cache_seq_indices

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
