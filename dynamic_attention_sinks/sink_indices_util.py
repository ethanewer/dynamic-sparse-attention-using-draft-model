import torch
from torch import Tensor
import numpy as np
from numba import njit  # type: ignore


@njit
def get_sink_indices_numba(
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
    sink_indices = get_sink_indices_numba(
        sorted_indices=sorted_indices,
        k=k,
        block_size=block_size,
        batch_size=batch_size,
        input_len=input_len,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
    )
    return torch.from_numpy(sink_indices)
