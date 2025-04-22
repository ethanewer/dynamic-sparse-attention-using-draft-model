import math
from typing import Literal

import torch
import torch.nn.functional as F
from minference import vertical_slash_sparse_attention  # type: ignore
from torch import Tensor
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention


def vertical_slash_sparse_attention_forward(
    module: LlamaAttention | Qwen2Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    v_idx: Tensor,
    s_idx: Tensor,
) -> tuple[torch.Tensor, None]:
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
        v_idx = repeat_kv(v_idx[..., None], module.num_key_value_groups)[..., 0]
        s_idx = repeat_kv(s_idx[..., None], module.num_key_value_groups)[..., 0]

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    attention_output = vertical_slash_sparse_attention(
        query, key, value, v_idx.clone(), s_idx.clone()
    )
    attention_output = attention_output.transpose(1, 2).contiguous()

    return attention_output, None


def compress_kv(
    key_states: Tensor,
    query_states: Tensor,
    value_states: Tensor,
    window_size: int,
    max_capacity_prompt: int,
    num_vertical: int,
    query_aggregation: Literal["mean", "max"],
    pooling: Literal["mean", "max"],
    kernel_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    _, _, seq_len, head_dim = query_states.shape
    assert key_states.shape[-2] == seq_len
    assert seq_len > max_capacity_prompt

    num_key_value_heads = key_states.shape[1]
    num_key_value_groups = query_states.shape[1] // num_key_value_heads

    attention_scores = torch.matmul(
        query_states[..., -window_size:, :],
        repeat_kv(key_states, num_key_value_groups).transpose(2, 3),
    ) / math.sqrt(head_dim)

    attention_mask = torch.full(
        (window_size, window_size),
        torch.finfo(attention_scores.dtype).min,
        device=attention_scores.device,
    ).triu(1)

    attention_scores[:, :, -window_size:, -window_size:] += attention_mask
    attention_scores = attention_scores.softmax(dim=-1, dtype=torch.float32)
    attention_scores = attention_scores[:, :, -window_size:, :-window_size]

    attention_scores = attention_scores.view(
        attention_scores.shape[0],
        num_key_value_heads,
        num_key_value_groups * attention_scores.shape[2],
        attention_scores.shape[3],
    )
    if query_aggregation == "mean":
        attention_scores = attention_scores.mean(dim=2)
    elif query_aggregation == "max":
        attention_scores = attention_scores.max(dim=2).values
    else:
        raise ValueError(f"{query_aggregation=} not supported.")

    if pooling == "mean":
        attention_scores = F.avg_pool1d(
            attention_scores,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )
    elif pooling == "max":
        attention_scores = F.max_pool1d(
            attention_scores,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )
    else:
        raise ValueError(f"{pooling=} not supported.")

    indices = attention_scores.topk(
        min(
            max(max_capacity_prompt - window_size, num_vertical),
            attention_scores.shape[-1],
        ),
        dim=-1,
    ).indices
    topk_indices = indices[..., : max_capacity_prompt - window_size]
    gather_indices = topk_indices[..., None].expand(-1, -1, -1, head_dim)

    key_past_compress = key_states[:, :, :-window_size, :].gather(
        dim=2,
        index=gather_indices,
    )
    value_past_compress = value_states[:, :, :-window_size, :].gather(
        dim=2,
        index=gather_indices,
    )
    key_window = key_states[:, :, -window_size:, :]
    value_window = value_states[:, :, -window_size:, :]
    key_states = torch.cat([key_past_compress, key_window], dim=2)
    value_states = torch.cat([value_past_compress, value_window], dim=2)

    return key_states, value_states, indices
