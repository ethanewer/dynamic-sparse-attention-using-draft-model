from typing import Any

import torch
from torch import Tensor
from transformers import (  # type: ignore
    DynamicCache,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
)

from .token_dropping_cache import TokenDroppingCache


def generate_reduced_attentions(
    model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    generation_kwargs: dict[str, Any] = {},
) -> tuple[Tensor, Tensor]:
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

    attention_scores: list[Tensor] = [
        torch.cat([a[0, :, :, :input_len].cpu() for a in attentions])
        .square()
        .sum(dim=0)
        for attentions in outputs.attentions  # type: ignore
    ]

    del outputs

    attention_matrix = torch.cat(attention_scores, dim=0)
    print(input_len, sequences.shape[1] - input_len, attention_matrix.shape)
    reduced_attentions = torch.cat(attention_scores, dim=0).sum(dim=0).sqrt()
    return sequences, reduced_attentions


def dynamic_attention_sinks_generate(
    model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    generated_ids: Tensor,
    reduced_attentions: Tensor,
    block_size: int,
    k: int,
    generation_kwargs: dict[str, Any] = {},
) -> Tensor:
    assert input_ids.shape[0] == 1
    input_len = input_ids.shape[1]
    position_ids = torch.arange(input_ids, device=input_ids.device)[None]  # type: ignore

    k = min(k, input_len - block_size)
    sink_indices = reduced_attentions[: input_len - block_size].topk(k).indices.tolist()
    cache_seq_indices = []
    past_key_values = TokenDroppingCache()

    for i in range(0, input_ids.shape[1] - 1, block_size):
        j = min(i + block_size, input_ids.shape[1] - 1)
        block_input_ids = input_ids[:, i:j]
        block_position_ids = position_ids[:, i:j]

        with torch.no_grad():
            _ = model(
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

    cache_size = min(block_size + k, input_len - 1)
    assert past_key_values.get_seq_length() == cache_size

    generated_ids = model.generate(  # type: ignore
        input_ids=input_ids[:, -cache_size - 1 :],
        attention_mask=torch.ones_like(input_ids),
        use_cache=True,
        past_key_values=past_key_values,
        **generation_kwargs,
    )

    return torch.cat((input_ids[:, : -cache_size - 1], generated_ids), dim=1)
