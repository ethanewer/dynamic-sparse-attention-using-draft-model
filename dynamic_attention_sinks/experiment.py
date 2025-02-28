from typing import Callable

import torch
from torch import Tensor
from transformers import LlamaForCausalLM, Qwen2ForCausalLM  # type: ignore
from transformers.models.llama.modeling_llama import LlamaAttention  # type: ignore
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention  # type: ignore

from .token_dropping_cache import TokenDroppingCache
from .sink_indices_util import get_sink_indices, update_indices

from tqdm.notebook import trange  # type: ignore

import time


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

    # sink_indices: list[list[list[int]]] = [
    #     get_sink_indices_for_single_head(
    #         fully_reduced_attention[batch_idx],
    #         k=k,
    #         block_size=block_size,
    #         input_len=input_len,
    #     )
    #     for batch_idx in range(input_ids.shape[0])
    # ]

    sink_indices = get_sink_indices(
        reduced_attentions.sum(dim=[0, 2])[None, :, None],
        k=k,
        block_size=block_size,
    )[0, :, 0]

    # import matplotlib.pyplot as plt

    # mask_ = torch.zeros(input_len, input_len)
    # for i_ in range(0, input_len, block_size):
    #     mask_[i_ : i_ + 2 * block_size, i_ : i_ + block_size] = 1

    # for i_, indices_ in enumerate(sink_indices[0][:-1]):
    #     print(indices_)
    #     mask_[(i_ + 1) * block_size : (i_ + 2) * block_size, indices_] = 1

    # mask_.tril_()
    # plt.imshow(mask_, cmap="gray_r", extent=(0, mask_.shape[1], 0, mask_.shape[0]))
    # plt.xticks(
    #     torch.arange(0, mask_.shape[1] + 1, 1), labels=" " * (mask_.shape[1] + 1)
    # )
    # plt.yticks(
    #     torch.arange(0, mask_.shape[0] + 1, 1), labels=" " * (mask_.shape[0] + 1)
    # )
    # plt.grid()
    # plt.show()

    # past_key_values = TokenDroppingCache()
    # cache_seq_indices: list[list[int]] = [[] for _ in range(input_ids.shape[0])]

    # for block_idx in range((input_len + block_size - 1) // block_size):
    #     block_start = block_idx * block_size
    #     block_end = min((block_idx + 1) * block_size, input_ids.shape[1])
    #     print(f"{block_idx=}, [{block_start}, {block_end}]")
    #     block_input_ids = input_ids[:, block_start:block_end]
    #     block_position_ids = position_ids[:, block_start:block_end]

    #     with torch.no_grad():
    #         outputs = model(
    #             input_ids=block_input_ids,
    #             position_ids=block_position_ids,
    #             use_cache=True,
    #             past_key_values=past_key_values,
    #         )

    #     selected_indices: list[list[int]] = [[]]
    #     new_cache_seq_indices: list[list[int]] = [[]]
    #     for batch_idx in range(input_ids.shape[0]):
    #         cache_seq_indices[batch_idx] += list(range(block_start, block_end))

    #         for cache_idx, seq_idx in enumerate(cache_seq_indices[batch_idx]):
    #             if (
    #                 seq_idx in sink_indices[batch_idx][block_idx]
    #                 or seq_idx >= block_end - block_size
    #             ):
    #                 selected_indices[batch_idx].append(cache_idx)
    #                 new_cache_seq_indices[batch_idx].append(seq_idx)

    #     past_key_values.token_select_indices(
    #         torch.tensor(selected_indices, device=input_ids.device)
    #     )
    #     cache_seq_indices = new_cache_seq_indices

    # print(min(cache_seq_indices[0]), max(cache_seq_indices[0]))

    past_key_values = TokenDroppingCache()
    cache_seq_indices: list[list[int]] = [[] for _ in range(input_ids.shape[0])]

    for block_idx in trange(
        (input_len + block_size - 1) // block_size, desc="block prefill"
    ):
        block_start = block_idx * block_size
        block_end = min((block_idx + 1) * block_size, input_len)
        block_input_ids = input_ids[:, block_start:block_end]
        block_position_ids = position_ids[:, block_start:block_end]
        t0 = time.time()

        with torch.no_grad():
            outputs = model(
                input_ids=block_input_ids,
                position_ids=block_position_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )

        t1 = time.time()

        selected_indices: list[list[int]] = [[] for _ in range(input_ids.shape[0])]
        new_cache_seq_indices: list[list[int]] = [[] for _ in range(input_ids.shape[0])]
        for batch_idx in range(input_ids.shape[0]):
            indices = sink_indices[batch_idx][block_idx]

            cache_seq_indices[batch_idx] += list(range(block_start, block_end))
            for cache_idx, seq_idx in enumerate(cache_seq_indices[batch_idx]):
                if seq_idx in indices or seq_idx >= block_end - block_size:
                    selected_indices[batch_idx].append(cache_idx)
                    new_cache_seq_indices[batch_idx].append(seq_idx)

        t2 = time.time()

        past_key_values.token_select_indices(
            torch.tensor(selected_indices, device=input_ids.device)
        )

        t3 = time.time()
        print(
            f"(v1) model forward: {t1 - t0:.3f}s, update indices: {t2 - t1:.3f}s, trim cache: {t3 - t2:.3f}s,"
        )

        cache_seq_indices = new_cache_seq_indices

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


def make_4d_list(dim1: int, dim2: int, dim3: int) -> list[list[list[list]]]:
    return [[[[] for _ in range(dim3)] for _ in range(dim2)] for _ in range(dim1)]


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

    sink_indices = get_sink_indices(reduced_attentions, k=k, block_size=block_size)

    past_key_values = TokenDroppingCache()

    cache_seq_indices = torch.empty(
        model.config.num_hidden_layers,
        input_ids.shape[0],
        model.config.num_key_value_heads,
        0,
        dtype=torch.int64,
    )

    for block_idx in trange(
        (input_len + block_size - 1) // block_size, desc="block prefill v2"
    ):
        block_start = block_idx * block_size
        block_end = min((block_idx + 1) * block_size, input_ids.shape[1])
        block_input_ids = input_ids[:, block_start:block_end]
        block_position_ids = position_ids[:, block_start:block_end]
        t0 = time.time()

        with torch.no_grad():
            _ = model(
                input_ids=block_input_ids,
                position_ids=block_position_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )

        t1 = time.time()

        selected_indices, cache_seq_indices = update_indices(
            sink_indices=sink_indices,
            cache_seq_indices=cache_seq_indices,
            block_idx=block_idx,
            block_size=block_size,
            k=k,
            input_len=input_len,
        )

        selected_indices = selected_indices.to(input_ids.device)

        t2 = time.time()
        print(block_size, k, block_idx, len(selected_indices[0][0][0]))

        for layer_idx in range(model.config.num_hidden_layers):
            past_key_values.token_select_indices(
                selected_indices[layer_idx],
                layer_idx=layer_idx,
            )

        t3 = time.time()
        print(
            f"(v2) model forward: {t1 - t0:.3f}s, update indices: {t2 - t1:.3f}s, trim cache: {t3 - t2:.3f}s,"
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
