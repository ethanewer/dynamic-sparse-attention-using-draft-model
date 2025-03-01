import torch
from torch import Tensor
import numpy as np
from numba import njit  # type: ignore


@njit
def get_sink_indices_np(
    sorted_indices: np.ndarray,
    k: int,
    block_size: int,
    batch_size: int,
    input_len: int,
    num_hidden_layers: int,
    num_key_value_heads: int,
) -> np.ndarray:
    sink_indices = -np.ones(
        (
            num_hidden_layers,
            batch_size,
            num_key_value_heads,
            (input_len + block_size - 1) // block_size,
            k,
        ),
        dtype=np.int64,
    )

    for layer_idx in range(num_hidden_layers):
        for batch_idx in range(batch_size):
            for head_idx in range(num_key_value_heads):
                for block_idx in range((input_len + block_size - 1) // block_size):
                    block_end = min((block_idx + 1) * block_size, input_len)
                    i = 0
                    for j in sorted_indices[layer_idx, batch_idx, head_idx]:
                        if j < block_end - block_size:
                            sink_indices[
                                layer_idx,
                                batch_idx,
                                head_idx,
                                block_idx,
                                i,
                            ] = j
                            i += 1
                            if i == k:
                                break

    return sink_indices


def get_sink_indices(
    reduced_attentions: Tensor,
    k: int,
    block_size: int,
) -> Tensor:
    num_hidden_layers, batch_size, num_key_value_heads, input_len = (
        reduced_attentions.shape
    )
    sorted_indices = reduced_attentions.argsort(dim=-1, descending=True).numpy()
    sink_indices = get_sink_indices_np(
        sorted_indices=sorted_indices,
        k=k,
        block_size=block_size,
        batch_size=batch_size,
        input_len=input_len,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
    )
    return torch.from_numpy(sink_indices)


@njit
def update_indices_np(
    sink_indices: np.ndarray,
    cache_seq_indices: np.ndarray,
    block_idx: int,
    block_size: int,
    k: int,
    input_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    block_start = block_idx * block_size
    block_end = min((block_idx + 1) * block_size, input_len)
    real_block_size = block_end - block_start

    num_hidden_layers, batch_size, num_key_value_heads = sink_indices.shape[:3]

    n_indices = min((block_idx + 1) * block_size, block_size + k)

    selected_indices = -np.ones(
        (
            num_hidden_layers,
            batch_size,
            num_key_value_heads,
            n_indices,
        ),
        dtype=np.int64,
    )

    new_cache_seq_indices = -np.ones(
        (
            num_hidden_layers,
            batch_size,
            num_key_value_heads,
            n_indices,
        ),
        dtype=np.int64,
    )

    for layer_idx in range(num_hidden_layers):
        for batch_idx in range(batch_size):
            for head_idx in range(num_key_value_heads):
                indices = sink_indices[layer_idx, batch_idx, head_idx, block_idx]
                seq_indices = cache_seq_indices[layer_idx, batch_idx, head_idx]

                mask_recent = seq_indices >= block_end - block_size
                mask_in_sink = np.isin(seq_indices, indices)

                valid_indices = np.where(mask_recent | mask_in_sink)[0]

                new_cache_idx = len(valid_indices)

                selected_indices[
                    layer_idx,
                    batch_idx,
                    head_idx,
                    :new_cache_idx,
                ] = valid_indices

                new_cache_seq_indices[
                    layer_idx,
                    batch_idx,
                    head_idx,
                    :new_cache_idx,
                ] = seq_indices[valid_indices]

    selected_indices[:, :, :, -real_block_size:] = np.arange(
        cache_seq_indices.shape[3],
        cache_seq_indices.shape[3] + real_block_size,
    )

    new_cache_seq_indices[:, :, :, -real_block_size:] = np.arange(
        block_start,
        block_end,
    )

    return selected_indices, new_cache_seq_indices


def update_indices(
    sink_indices: Tensor,
    cache_seq_indices: Tensor,
    block_idx: int,
    block_size: int,
    k: int,
    input_len: int,
) -> tuple[Tensor, Tensor]:
    selected_indices, new_cache_seq_indices = update_indices_np(
        sink_indices=sink_indices.numpy(),
        cache_seq_indices=cache_seq_indices.numpy(),
        block_idx=block_idx,
        block_size=block_size,
        k=k,
        input_len=input_len,
    )
    return torch.from_numpy(selected_indices), torch.from_numpy(new_cache_seq_indices)


def get_cache_update_indices(
    reduced_attentions: Tensor,
    k: int,
    block_size: int,
    reduce_heads: bool = False,
    device: str = "cpu",
) -> list[Tensor]:
    num_hidden_layers, batch_size, num_key_value_heads, input_len = (
        reduced_attentions.shape
    )

    sorted_indices = reduced_attentions.argsort(dim=-1, descending=True).numpy()
    sink_indices = get_sink_indices_np(
        sorted_indices=sorted_indices,
        k=k,
        block_size=block_size,
        batch_size=batch_size,
        input_len=input_len,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
    )

    cache_seq_indices = np.zeros(
        (num_hidden_layers, input_len, num_key_value_heads, 0),
        dtype=np.int64,
    )

    cache_update_indices = []

    for block_idx in range((input_len + block_size - 1) // block_size):
        block_selected_indices, cache_seq_indices = update_indices_np(
            sink_indices=sink_indices,
            cache_seq_indices=cache_seq_indices,
            block_idx=block_idx,
            block_size=block_size,
            k=k,
            input_len=input_len,
        )

        if reduce_heads:
            assert (
                block_selected_indices.shape[0] == 1
                and block_selected_indices.shape[2] == 1
            )
            block_selected_indices = block_selected_indices[0, :, 0]

        cache_update_indices.append(torch.from_numpy(block_selected_indices).to(device))

    return cache_update_indices
