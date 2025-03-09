import os
import time
from typing import Optional

import psutil  # type: ignore
import torch
import torch.nn.functional as F
from torch import Tensor, dtype
from torch.types import Device
from transformers.models.llama.modeling_llama import LlamaAttention  # type: ignore
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention  # type: ignore


def log_metrics(stage, start_time, start_mem):
    end_time = time.time()
    if torch.cuda.is_available():
        end_mem = torch.cuda.memory_allocated()
    else:
        process = psutil.Process(os.getpid())
        end_mem = process.memory_info().rss

    print(f"{stage:32} | {end_time - start_time:.2e}s, {end_mem - start_mem:.2e} bytes")
    return end_time, end_mem


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


def unstack_attn(attn_output: Tensor, batch_size: int, seq_len: int) -> Tensor:
    total_batch_size, num_key_value_heads, block_size, head_dim = attn_output.shape
    num_blocks = total_batch_size // batch_size
    return (
        attn_output.view(
            batch_size,
            num_blocks,
            num_key_value_heads,
            block_size,
            head_dim,
        )
        .transpose(1, 2)
        .reshape(
            batch_size,
            num_key_value_heads,
            num_blocks * block_size,
            head_dim,
        )[..., :seq_len, :]
        .transpose(1, 2)
        .contiguous()
    )


def make_causal_mask(
    indices: Tensor,
    batch_size: int,
    seq_len: int,
    block_size: int,
    dtype: dtype = torch.float32,
    device: Device = "cpu",
) -> Tensor:
    pad = -seq_len % block_size
    causal_mask = torch.full(
        (batch_size, 1, seq_len + pad, seq_len + 1),
        fill_value=torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    ).triu_(1)

    assert causal_mask is not None
    assert causal_mask.shape[-2] % block_size == 0, causal_mask.shape

    mask_expanded_indices = indices[
        : causal_mask.shape[0], : causal_mask.shape[1], :, None, :
    ].expand(-1, -1, -1, block_size, -1)

    causal_mask = causal_mask.view(
        causal_mask.shape[0],
        causal_mask.shape[1],
        (seq_len + pad) // block_size,
        block_size,
        -1,
    ).gather(dim=4, index=mask_expanded_indices)

    causal_mask = stack_block_along_batch(causal_mask)
    return causal_mask


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
    batch_size = query.shape[0]
    origional_seq_len = query.shape[-2]

    causal_mask = attention_mask

    pad = -query.shape[-2] % block_size
    if pad > 0:
        query = F.pad(query, (0, 0, 0, pad))

    query = query.view(
        query.shape[0],
        query.shape[1],
        query.shape[2] // block_size,
        block_size,
        query.shape[3],
    )

    kv_expanded_indices = indices.view(
        indices.shape[0],
        indices.shape[1],
        -1,
        1,
    ).expand(-1, -1, -1, key.shape[3])

    key = key.gather(
        dim=2,
        index=kv_expanded_indices,
    ).view(key.shape[0], key.shape[1], indices.shape[2], indices.shape[3], key.shape[3])

    value = value.gather(
        dim=2,
        index=kv_expanded_indices,
    ).view(
        value.shape[0],
        value.shape[1],
        indices.shape[2],
        indices.shape[3],
        value.shape[3],
    )

    del indices, kv_expanded_indices

    if hasattr(module, "num_key_value_groups"):
        num_key_value_groups = module.num_key_value_groups
    else:
        num_key_value_groups = 1

    query = stack_block_along_batch(query)
    key = stack_block_along_batch(key, num_key_value_groups)
    value = stack_block_along_batch(value, num_key_value_groups)

    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
    )

    attn_output = unstack_attn(attn_output, batch_size, origional_seq_len)

    return attn_output, None
