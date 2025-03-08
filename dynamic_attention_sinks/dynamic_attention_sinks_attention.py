from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.models.llama.modeling_llama import LlamaAttention  # type: ignore
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention  # type: ignore


def repeat_block(block: Tensor, num_repeat: int) -> Tensor:
    batch_size, num_key_value_heads, num_blocks, block_dim, head_dim = block.shape
    if num_repeat == 1:
        return block

    block = block[:, :, None, :, :].expand(
        batch_size,
        num_key_value_heads,
        num_repeat,
        num_blocks,
        block_dim,
        head_dim,
    )
    return block.reshape(
        batch_size,
        num_key_value_heads * num_repeat,
        num_blocks,
        block_dim,
        head_dim,
    )


def stack_block_along_batch(block: Tensor, num_key_value_groups: int = 1) -> Tensor:
    batch_size, num_heads, num_blocks, block_size, head_dim = block.shape
    if num_key_value_groups == 1:
        stacked_block = block.transpose(1, 2).reshape(
            batch_size * num_blocks,
            num_heads,
            block_size,
            head_dim,
        )
    else:
        stacked_block = (
            block.transpose(1, 2)[:, :, :, None, :, :]
            .expand(
                batch_size,
                num_blocks,
                num_heads,
                num_key_value_groups,
                block_size,
                head_dim,
            )
            .reshape(
                batch_size * num_blocks,
                num_key_value_groups * num_heads,
                block_size,
                head_dim,
            )
        )

    if not stacked_block.is_contiguous():
        stacked_block = stacked_block.contiguous()

    return stacked_block


def unstack_block_along_batch(block: Tensor, batch_size: int) -> Tensor:
    total_batch_size, num_key_value_heads, block_size, head_dim = block.shape
    num_blocks = total_batch_size // batch_size
    return block.view(
        batch_size,
        num_blocks,
        num_key_value_heads,
        block_size,
        head_dim,
    ).transpose(1, 2)


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
    # print(f"{query.numel()=}, {key.numel()=}, {value.numel()=}")

    assert query.shape[-2] == key.shape[-2] and key.shape[-2] == value.shape[-2], (
        query.shape,
        key.shape,
        value.shape,
    )

    batch_size = query.shape[0]
    origional_seq_len = query.shape[-2]

    pad = -query.shape[-2] % block_size
    if pad > 0:
        query = F.pad(query, (0, 0, 0, pad))

    causal_mask = attention_mask
    if causal_mask is not None:
        causal_mask = F.pad(
            causal_mask,
            (0, pad, 0, 1),
            value=torch.finfo(query.dtype).min,
        ).expand(query.shape[0], -1, -1, -1)
    else:
        causal_mask = torch.full(
            (query.shape[0], 1, origional_seq_len + pad, origional_seq_len + 1),
            fill_value=torch.finfo(query.dtype).min,
            dtype=query.dtype,
            device=query.device,
        ).triu_(1)

    assert causal_mask is not None
    assert query.shape[-2] % block_size == 0, query.shape
    assert causal_mask.shape[-2] % block_size == 0, causal_mask.shape

    block_query = query.view(
        query.shape[0],
        query.shape[1],
        query.shape[2] // block_size,
        block_size,
        query.shape[3],
    )

    del query

    kv_expanded_indices = (
        indices.view(indices.shape[0], indices.shape[1], -1, 1)
        .expand(-1, -1, -1, key.shape[3])
        .relu()
    )

    block_key = key.gather(
        dim=2,
        index=kv_expanded_indices,
    ).view(key.shape[0], key.shape[1], indices.shape[2], indices.shape[3], key.shape[3])

    block_value = value.gather(
        dim=2,
        index=kv_expanded_indices,
    ).view(
        value.shape[0],
        value.shape[1],
        indices.shape[2],
        indices.shape[3],
        value.shape[3],
    )

    del key, value, kv_expanded_indices

    mask_expanded_indices = (
        indices[: causal_mask.shape[0], : causal_mask.shape[1], :, None, :]
        .expand(-1, -1, -1, block_size, -1)
        .relu()
    )

    block_mask = causal_mask.view(
        causal_mask.shape[0],
        causal_mask.shape[1],
        block_query.shape[2],
        block_query.shape[3],
        -1,
    ).gather(dim=4, index=mask_expanded_indices)

    del causal_mask, mask_expanded_indices

    # print(f"{block_query.numel()=}, {block_key.numel()=}, {block_value.numel()=}")

    if hasattr(module, "num_key_value_groups"):
        num_key_value_groups = module.num_key_value_groups
    else:
        num_key_value_groups = 1

    block_query = stack_block_along_batch(block_query)
    block_key = stack_block_along_batch(block_key, num_key_value_groups)
    block_value = stack_block_along_batch(block_value, num_key_value_groups)
    block_mask = stack_block_along_batch(block_mask)

    # print(f"{block_query.numel()=}, {block_key.numel()=}, {block_value.numel()=}\n")

    attn_output = F.scaled_dot_product_attention(
        block_query,
        block_key,
        block_value,
        attn_mask=block_mask,
        dropout_p=dropout,
        scale=scaling,
    )

    attn_output = unstack_block_along_batch(attn_output, batch_size=batch_size)

    attn_output = attn_output.reshape(
        attn_output.shape[0],
        attn_output.shape[1],
        attn_output.shape[2] * attn_output.shape[3],
        attn_output.shape[4],
    )
    attn_output = attn_output[..., :origional_seq_len, :]

    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
