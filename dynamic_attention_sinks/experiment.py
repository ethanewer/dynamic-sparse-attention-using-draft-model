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


def get_sink_indices_for_single_head(
    reduced_attentions: Tensor,
    k: int,
    block_size: int,
    input_len: int,
) -> list[list[int]]:
    assert reduced_attentions.shape == (input_len,)

    sink_indices: list[list[int]] = [
        reduced_attentions[: input_len - block_size].topk(k).indices.tolist()
    ]
    for i in range((input_len + block_size - 1) // block_size - 2, -1, -1):
        prev_indices = [j for j in sink_indices[-1] if j in range(i * block_size)]
        indices = [j for j in range(i * block_size) if j not in sink_indices[-1]]
        if indices:
            a = reduced_attentions[indices].square()
            sink_indices.append(
                prev_indices
                + [indices[i] for i in a.topk(k - len(prev_indices)).indices]
            )
        else:
            sink_indices.append(prev_indices)

    return sink_indices[::-1]


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

    fully_reduced_attention = torch.cat(reduced_attentions, dim=1).sum(dim=1)
    sink_indices: list[list[list[int]]] = [
        get_sink_indices_for_single_head(
            fully_reduced_attention[batch_idx],
            k=k,
            block_size=block_size,
            input_len=input_len,
        )
        for batch_idx in range(input_ids.shape[0])
    ]

    print(sink_indices)
    # import matplotlib.pyplot as plt

    # mask = torch.zeros(input_len, input_len)
    # for i_ in range(0, input_len, block_size):
    #     mask[i_ : i_ + 2 * block_size, i_ : i_ + block_size] = 1

    # for i_, indices_ in enumerate(sink_indices[0][:-1]):
    #     j_ = (i_ + 1) * block_size
    #     k_ = (i_ + 2) * block_size
    #     mask[j_:k_, indices_] = 1

    # mask.tril_()
    # plt.imshow(mask, cmap="gray_r", extent=(0, mask.shape[1], 0, mask.shape[0]))
    # plt.xticks(torch.arange(0, mask.shape[1] + 1, 1), labels=" " * (mask.shape[1] + 1))
    # plt.yticks(torch.arange(0, mask.shape[0] + 1, 1), labels=" " * (mask.shape[0] + 1))
    # plt.grid()
    # plt.show()

    past_key_values = TokenDroppingCache()
    cache_seq_indices: list[list[int]] = [[] for _ in range(input_ids.shape[0])]

    for block_idx in range((input_len + block_size - 1) // block_size - 2):
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

        selected_indices: list[list[int]] = [[] for _ in range(input_ids.shape[0])]
        new_cache_seq_indices: list[list[int]] = [[] for _ in range(input_ids.shape[0])]
        for batch_idx in range(input_ids.shape[0]):
            indices = sink_indices[batch_idx][block_idx]

            cache_seq_indices[batch_idx] += list(range(block_start, block_end))
            for cache_idx, seq_idx in enumerate(cache_seq_indices[batch_idx]):
                if seq_idx in indices or seq_idx >= block_end - block_size:
                    selected_indices[batch_idx].append(cache_idx)
                    new_cache_seq_indices[batch_idx].append(seq_idx)

        print(cache_seq_indices)

        past_key_values.token_select_indices(
            torch.tensor(selected_indices, device=input_ids.device)
        )

        cache_seq_indices = new_cache_seq_indices
        # selected_indices: list[list[int]] = [[]]
        # new_cache_seq_indices: list[list[int]] = [[]]
        # for batch_idx in range(input_ids.shape[0]):
        #     cache_seq_indices[batch_idx] += list(range(block_start, block_end))

        #     for cache_idx, seq_idx in enumerate(cache_seq_indices[batch_idx]):
        #         if (
        #             seq_idx in sink_indices[batch_idx]
        #             or seq_idx >= block_end - block_size
        #         ):
        #             selected_indices[batch_idx].append(cache_idx)
        #             new_cache_seq_indices[batch_idx].append(seq_idx)

        # past_key_values.token_select_indices(
        #     torch.tensor(selected_indices, device=input_ids.device)
        # )
        # cache_seq_indices = new_cache_seq_indices

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

    sink_indices: list[list[list[list[list[int]]]]] = []
    for layer_idx in range(model.config.num_hidden_layers):
        sink_indices.append([])
        for batch_idx in range(input_ids.shape[0]):
            sink_indices[layer_idx].append([])
            for head_idx in range(model.config.num_key_value_heads):
                sink_indices[layer_idx][batch_idx].append(
                    get_sink_indices_for_single_head(
                        reduced_attentions[layer_idx][batch_idx, head_idx],
                        k=k,
                        block_size=block_size,
                        input_len=input_len,
                    )
                )

    print(sink_indices[0][0][0])
    print(sink_indices[-1][-1][-1])

    past_key_values = TokenDroppingCache()
    cache_seq_indices: list[list[list[list[int]]]] = make_4d_list(
        model.config.num_hidden_layers,
        input_ids.shape[0],
        model.config.num_key_value_heads,
    )

    for block_idx in range((input_len + block_size - 1) // block_size - 2):
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

        selected_indices = make_4d_list(
            model.config.num_hidden_layers,
            input_ids.shape[0],
            model.config.num_key_value_heads,
        )
        new_cache_seq_indices = make_4d_list(
            model.config.num_hidden_layers,
            input_ids.shape[0],
            model.config.num_key_value_heads,
        )
        for layer_idx in range(model.config.num_hidden_layers):
            for batch_idx in range(input_ids.shape[0]):
                for head_idx in range(model.config.num_key_value_heads):
                    # print(layer_idx, batch_idx, head_idx, block_idx)
                    # print(len(sink_indices))
                    # print(len(sink_indices[layer_idx]))
                    # print(len(sink_indices[layer_idx][batch_idx]))
                    # print(len(sink_indices[layer_idx][batch_idx][block_idx]))

                    indices = sink_indices[layer_idx][batch_idx][head_idx][block_idx]

                    cache_seq_indices[layer_idx][batch_idx][head_idx] += list(
                        range(block_start, block_end)
                    )
                    for cache_idx, seq_idx in enumerate(
                        cache_seq_indices[layer_idx][batch_idx][head_idx]
                    ):
                        if seq_idx in indices or seq_idx >= block_end - block_size:
                            selected_indices[layer_idx][batch_idx][head_idx].append(
                                cache_idx
                            )
                            new_cache_seq_indices[layer_idx][batch_idx][
                                head_idx
                            ].append(seq_idx)

            past_key_values.token_select_indices(
                torch.tensor(selected_indices[layer_idx], device=input_ids.device)
            )
        print(cache_seq_indices[-1][-1][-1])
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
