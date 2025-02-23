from typing import Callable

import torch
from torch import Tensor

from .token_dropping_cache import TokenDroppingCache


def streaming_llm_experiment(
    model: Callable,
    input_ids: Tensor,
    generated_ids: Tensor,
    block_size: int,
    k: int,
) -> Tensor:
    assert input_ids.shape[0] == 1
    input_len = input_ids.shape[1]
    position_ids = torch.arange(input_len, device=input_ids.device)[None]

    k = min(k, input_len - block_size)
    sink_indices = list(range(k))
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

    assert past_key_values.get_seq_length() == min(block_size + k, input_len)

    logits = []
    for i in range(input_len + 1, generated_ids.shape[1]):
        input_ids = generated_ids[:, i - 1 : i]
        position_ids = torch.tensor([[i - 1]], device=input_ids.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
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
                past_key_values=past_key_values,
                use_cache=True,
            )

        logits.append(outputs.logits[0, -1:])

    return torch.cat(logits).float().cpu()
