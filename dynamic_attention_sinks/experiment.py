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
    reduced_attentions: list[Tensor],
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

    fully_reduced_attentions = torch.cat(reduced_attentions, dim=1).sum(dim=1)

    sink_indices: list[list[int]] = (
        fully_reduced_attentions[:, : input_len - block_size]
        .topk(k, dim=1)
        .indices.tolist()
    )
    cache_seq_indices: list[list[int]] = [[] for _ in range(input_ids.shape[0])]
    past_key_values = TokenDroppingCache()

    for block_start in range(0, input_ids.shape[1], block_size):
        block_end = min(block_start + block_size, input_ids.shape[1])
        block_input_ids = input_ids[:, block_start:block_end]
        block_position_ids = position_ids[:, block_start:block_end]

        with torch.no_grad():
            outputs = model(
                input_ids=block_input_ids,
                position_ids=block_position_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )

        selected_indices: list[list[int]] = [[]]
        new_cache_seq_indices: list[list[int]] = [[]]
        for batch_idx in range(input_ids.shape[0]):
            cache_seq_indices[batch_idx] += list(range(block_start, block_end))

            for cache_idx, seq_idx in enumerate(cache_seq_indices[batch_idx]):
                if (
                    seq_idx in sink_indices[batch_idx]
                    or seq_idx >= block_end - block_size
                ):
                    selected_indices[batch_idx].append(cache_idx)
                    new_cache_seq_indices[batch_idx].append(seq_idx)

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
    reduced_attentions: list[Tensor],
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
    sink_indices: list[list[list[list[list[int]]]]] = [
        [
            a[: input_len - block_size].topk(k, dim=2).indices.tolist()
            for a in reduced_attentions
        ]
    ]

    for block_idx in range((input_len + block_size - 1) // block_size - 2, -1, -1):
        block_start = block_idx + block_size
        block_end = min((block_idx + 1) * block_size, input_ids.shape[1])
        new_indices: list[list[list[list[int]]]] = [
            [[] for _ in range(input_ids.shape[0])]
            for _ in range(model.config.num_hidden_layers)
        ]
        for layer_idx in range(model.config.num_hidden_layers):
            for batch_idx in range(input_ids.shape[0]):
                for head_idx in range(model.config.num_key_value_heads):
                    prev_indices = [
                        j
                        for j in sink_indices[-1][layer_idx][batch_idx][head_idx]
                        if j in list(range(block_idx * block_size))
                    ]
                    indices = [
                        j
                        for j in range(block_idx * block_size)
                        if j not in sink_indices[-1][layer_idx][batch_idx][head_idx]
                    ]
                    if indices:
                        a = reduced_attentions[layer_idx][batch_idx, head_idx, indices]
                        new_indices[layer_idx][batch_idx].append(
                            prev_indices
                            + [
                                indices[block_idx]
                                for block_idx in a.topk(k - len(prev_indices)).indices
                            ]
                        )
                    else:
                        new_indices[layer_idx][batch_idx].append(prev_indices)

        sink_indices.append(new_indices)

    # validate sink indices
    for block_idx in range((input_len + block_size - 1) // block_size - 2, -1, -1):
        block_start = block_idx * block_size
        block_end = min((block_idx + 1) * block_size, input_ids.shape[1])

        l1 = len(sink_indices[0][0])
        for layer_idx in range(model.config.num_hidden_layers):
            assert len(sink_indices[block_idx][layer_idx]) == l1, (
                layer_idx,
                len(sink_indices[block_idx][layer_idx]),
                l1,
            )
            l2 = len(sink_indices[0][0][0])
            for batch_idx in range(input_ids.shape[0]):
                assert len(sink_indices[block_idx][layer_idx][batch_idx]) == l2, (
                    layer_idx,
                    len(sink_indices[block_idx][layer_idx]),
                    l1,
                )
                l3 = len(sink_indices[0][0][0][0])
                for head_idx in range(model.config.num_key_value_heads):
                    # print(l1, l2, l3, "    ", layer_idx, batch_idx, head_idx)
                    assert (
                        len(sink_indices[block_idx][layer_idx][batch_idx][head_idx])
                        == l3
                    ), (
                        layer_idx,
                        len(sink_indices[block_idx][layer_idx]),
                        l1,
                    )
                    for i in sink_indices[block_idx][layer_idx][batch_idx][head_idx]:
                        assert i < block_end, (
                            sink_indices[block_idx][layer_idx][batch_idx][head_idx],
                            block_start,
                            block_end,
                        )
    ########################

    sink_indices = sink_indices[::-1]

    cache_seq_indices: list[list[list[list[int]]]] = [
        [
            [[] for _ in range(model.config.num_key_value_heads)]
            for _ in range(input_ids.shape[0])
        ]
        for _ in range(model.config.num_hidden_layers)
    ]
    past_key_values = TokenDroppingCache()

    for block_idx, block_sink_indices in enumerate(sink_indices):
        block_start = block_idx * block_size
        block_end = min((block_idx + 1) * block_size, input_ids.shape[1])
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
            selected_indices: list[list[list[int]]] = [
                [[] for _ in range(model.config.num_key_value_heads)]
                for _ in range(input_ids.shape[0])
            ]
            new_cache_seq_indices: list[list[list[int]]] = [
                [[] for _ in range(model.config.num_key_value_heads)]
                for _ in range(input_ids.shape[0])
            ]

            for batch_idx in range(input_ids.shape[0]):
                for head_idx in range(model.config.num_key_value_heads):
                    for cache_idx, seq_idx in enumerate(
                        cache_seq_indices[layer_idx][batch_idx][head_idx]
                        + list(range(block_start, block_end))
                    ):
                        if (
                            seq_idx
                            in block_sink_indices[layer_idx][batch_idx][head_idx]
                            or seq_idx >= block_end - block_size
                        ):
                            selected_indices[batch_idx][head_idx].append(cache_idx)
                            new_cache_seq_indices[batch_idx][head_idx].append(seq_idx)

            past_key_values.token_select_indices(
                torch.tensor(selected_indices, device=input_ids.device)
            )

            cache_seq_indices[layer_idx] = new_cache_seq_indices

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
