from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.models.llama.modeling_llama import LlamaAttention  # type: ignore
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention  # type: ignore


def repeat_block(block: Tensor, n_rep: int) -> Tensor:
    b, num_key_value_heads, m, n, d = block.shape
    if n_rep == 1:
        return block

    block = block[:, :, None, :, :].expand(b, num_key_value_heads, n_rep, m, n, d)
    return block.reshape(b, num_key_value_heads * n_rep, m, n, d)


def dynamic_attention_sinks_attention_forward(
    module: LlamaAttention | Qwen2Attention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
    block_size: int,
    indices: Tensor,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
) -> tuple[Tensor, None]:
    assert query.shape[-2] == key.shape[-2] and key.shape[-2] == value.shape[-2], (
        query.shape,
        key.shape,
        value.shape,
    )

    causal_mask = attention_mask
    if causal_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]
    else:
        causal_mask = torch.ones(
            query.shape[-2],
            key.shape[-2],
            device=query.device,
        ).tril_()
        causal_mask = -3.4028e38 * (1 - causal_mask[None, None])

    assert causal_mask is not None
    causal_mask = causal_mask.expand(query.shape[0], query.shape[1], -1, -1)

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.

    origional_seq_len = query.shape[-2]

    pad = -query.shape[-2] % block_size
    if pad > 0:
        query = F.pad(query, (0, 0, 0, pad))
        key = F.pad(key, (0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, pad))
        causal_mask = F.pad(causal_mask, (0, pad, 0, pad), value=-3.4028e38)

    assert query.shape[-2] % block_size == 0, query.shape
    assert key.shape[-2] % block_size == 0, key.shape
    assert value.shape[-2] % block_size == 0, value.shape
    assert causal_mask.shape[-2] % block_size == 0, causal_mask.shape
    assert causal_mask.shape[-1] % block_size == 0, causal_mask.shape

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    kv_expanded_indices = (
        indices[:, :, :, :, None].expand(-1, -1, -1, -1, key.shape[-1]).relu()
    )

    mask_expanded_indices = (
        indices[:, :, :, None, :].expand(-1, -1, -1, block_size, -1).relu()
    )

    invalid_expanded_indices = (indices == -1)[:, :, :, None, :].expand(
        -1,
        -1,
        -1,
        block_size,
        -1,
    )

    block_query = query.view(
        query.shape[0],
        query.shape[1],
        query.shape[2] // block_size,
        block_size,
        query.shape[3],
    )

    block_key = (
        key[:, :, None]
        .expand(-1, -1, key.shape[2] // block_size, -1, -1)
        .gather(dim=3, index=kv_expanded_indices)
    )

    block_value = (
        value[:, :, None]
        .expand(-1, -1, value.shape[2] // block_size, -1, -1)
        .gather(dim=3, index=kv_expanded_indices)
    )

    block_mask = causal_mask.view(
        causal_mask.shape[0],
        causal_mask.shape[1],
        query.shape[2] // block_size,
        block_size,
        -1,
    ).gather(dim=4, index=mask_expanded_indices)

    block_mask = torch.where(invalid_expanded_indices, -3.4028e38, block_mask)

    if hasattr(module, "num_key_value_groups"):
        block_key = repeat_block(block_key, module.num_key_value_groups)
        block_value = repeat_block(block_value, module.num_key_value_groups)
        block_mask = repeat_block(block_mask, module.num_key_value_groups)

    attn_output = F.scaled_dot_product_attention(
        block_query,
        block_key,
        block_value,
        attn_mask=block_mask,
        dropout_p=dropout,
        scale=scaling,
    )

    attn_output = attn_output[..., :origional_seq_len, :]

    attn_output = attn_output.view(
        attn_output.shape[0],
        attn_output.shape[1],
        attn_output.shape[2] * attn_output.shape[3],
        attn_output.shape[4],
    )

    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
