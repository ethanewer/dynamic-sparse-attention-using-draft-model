import math
from typing import Optional

import torch
import torch.nn as nn
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

    attn_output = vertical_slash_sparse_attention(
        query, key, value, v_idx.clone(), s_idx.clone()
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


def pool_queries_for_gqa(hidden_states: Tensor, n_rep: int) -> Tensor:
    batch, num_attention_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    num_key_value_heads = num_attention_heads // n_rep
    return hidden_states.view(
        batch,
        num_key_value_heads,
        n_rep,
        seq_len,
        head_dim,
    ).mean(dim=2)


def compress_kv(
    key_states: Tensor,
    query_states: Tensor,
    value_states: Tensor,
    attention_mask: Optional[Tensor],
    window_size: int,
    max_capacity_prompt: int,
    pooling: str,
    kernel_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    _, _, seq_len, head_dim = query_states.shape
    assert key_states.shape[-2] == seq_len
    assert seq_len > max_capacity_prompt

    if query_states.shape[1] > key_states.shape[1]:
        queries_per_key_value = query_states.shape[1] // key_states.shape[1]
        query_states = pool_queries_for_gqa(
            query_states,
            queries_per_key_value,
        )
        assert query_states.shape[1] == key_states.shape[1]

    attention_weights = torch.matmul(
        query_states[..., -window_size:, :],
        key_states.transpose(2, 3),
    ) / math.sqrt(head_dim)
    mask = torch.full(
        (window_size, window_size),
        torch.finfo(attention_weights.dtype).min,
        device=attention_weights.device,
    )
    mask_cond = torch.arange(mask.size(-1), device=attention_weights.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(attention_weights.device)

    if attention_mask is not None:
        attention_mask = mask[None, None, :, :]
        attention_weights[:, :, -window_size:, -window_size:] += attention_mask

    attention_weights = nn.functional.softmax(
        attention_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    attention_weights_sum = attention_weights[:, :, -window_size:, :-window_size].sum(
        dim=-2
    )
    if pooling == "avgpool":
        attention_cache = F.avg_pool1d(
            attention_weights_sum,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )
    elif pooling == "maxpool":
        attention_cache = F.max_pool1d(
            attention_weights_sum,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )
    else:
        raise ValueError("Pooling method not supported")

    indices = attention_cache.topk(
        max_capacity_prompt - window_size,
        dim=-1,
    ).indices

    gather_indices = indices[..., None].expand(-1, -1, -1, head_dim)
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
