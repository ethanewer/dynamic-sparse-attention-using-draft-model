from typing import Optional

import mlx.core as mx
import numpy as np
from mlx_lm.models import llama, qwen2
from mlx_lm.models.base import create_causal_mask

from .token_dropping_cache import TokenDroppingKVCache


def streaming_llm_experiment(
    model: llama.Model | qwen2.Model,
    inputs: mx.array,
    generated_ids: mx.array,
    window_size: int,
    n_sinks: int,
    dtype=mx.bfloat16,
) -> mx.array:
    assert inputs.shape[0] == 1
    input_len = inputs.shape[1]

    mask_np = np.tril(np.ones((input_len, input_len))) - np.tril(
        np.ones((input_len, input_len)), -window_size
    )

    n_sinks = min(n_sinks, input_len)
    for i in range(n_sinks):
        mask_np[:, i] = 1

    mask = mx.array(-3.4028e38 * (1 - np.tril(mask_np)), dtype=dtype)

    cache = [TokenDroppingKVCache() for _ in range(len(model.model.layers))]
    _ = model(inputs=inputs, mask=mask, cache=cache)

    logits = []
    for i in range(input_len + 1, generated_ids.shape[1]):
        inputs = generated_ids[:, i - 1 : i]
        assert cache[0].offset == i - 1

        if cache[0].true_offset >= window_size + n_sinks:
            sink_indices = list(range(n_sinks))
            window_indices = list(
                range(
                    cache[0].true_offset - window_size + 1,
                    cache[0].true_offset,
                )
            )
            for c in cache:
                c.token_select_indices(sink_indices + window_indices)

        outputs = model(inputs=inputs, cache=cache)

        logits.append(outputs[0, -1:])

    return mx.concatenate(logits)


def create_attention_mask(
    inputs: mx.array,
    cache: Optional[list[TokenDroppingKVCache]],
) -> Optional[mx.array]:
    T = inputs.shape[1]
    if T > 1:
        window_size = None
        offset = 0
        if cache is not None and cache[0] is not None:
            c = cache[0]
            offset = c.true_offset
        mask = create_causal_mask(T, offset, window_size=window_size)
        mask = mask.astype(inputs.dtype)
    else:
        mask = None

    return mask


def dynamic_attention_sinks_experiment(
    model: llama.Model | qwen2.Model,
    inputs: mx.array,
    generated_ids: mx.array,
    reduced_attentions: mx.array,
    block_size: int,
    k: int,
) -> mx.array:
    assert inputs.shape[0] == 1
    input_len = inputs.shape[1]

    k = min(k, input_len - block_size)
    sink_indices = mx.argsort(-reduced_attentions[: input_len - block_size])[:k]
    cache_seq_indices = []
    cache = [TokenDroppingKVCache() for _ in range(len(model.model.layers))]

    for i in range(0, inputs.shape[1], block_size):
        j = min(i + block_size, inputs.shape[1])
        block_inputs = inputs[:, i:j]
        mask = create_attention_mask(block_inputs, cache)
        assert cache[0].offset == i

        outputs = model(inputs=block_inputs, mask=mask, cache=cache)  # type: ignore

        cache_seq_indices += list(range(i, j))
        selected_indices = []
        new_cache_seq_indices = []
        for cache_idx, seq_idx in enumerate(cache_seq_indices):
            if seq_idx in sink_indices or seq_idx >= j - block_size:  # type: ignore
                selected_indices.append(cache_idx)
                new_cache_seq_indices.append(seq_idx)

        for c in cache:
            c.token_select_indices(selected_indices)

        cache_seq_indices = new_cache_seq_indices

    assert cache[0].true_offset == min(block_size + k, input_len)

    logits = []
    for i in range(input_len + 1, generated_ids.shape[1]):
        inputs = generated_ids[:, i - 1 : i]
        assert cache[0].offset == i - 1

        outputs = model(inputs=inputs, cache=cache)

        logits.append(outputs[0, -1:])

    return mx.concatenate(logits)
