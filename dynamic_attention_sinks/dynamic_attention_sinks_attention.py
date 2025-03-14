from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, dtype
from torch.types import Device
from transformers.models.llama.modeling_llama import LlamaAttention  # type: ignore
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention  # type: ignore


def repeat_kv(hidden_states: Tensor, num_repeats: int) -> Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if num_repeats == 1:
        return hidden_states
    else:
        return (
            hidden_states[:, :, None, :, :]
            .expand(
                batch,
                num_key_value_heads,
                num_repeats,
                slen,
                head_dim,
            )
            .reshape(
                batch,
                num_key_value_heads * num_repeats,
                slen,
                head_dim,
            )
            .contiguous()
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


def das_attention_parallel_forward(
    module: LlamaAttention | Qwen2Attention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor,
    block_size: int,
    indices: Tensor,
    origional_seq_len: int,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
) -> tuple[Tensor, None]:
    batch_size = query.shape[0]

    query = query.view(
        query.shape[0],
        query.shape[1],
        query.shape[2] // block_size,
        block_size,
        query.shape[3],
    )

    expanded_indices = indices.view(
        indices.shape[0],
        indices.shape[1],
        -1,
        1,
    ).expand(-1, -1, -1, key.shape[3])

    key = key.gather(
        dim=2,
        index=expanded_indices,
    ).view(
        key.shape[0],
        key.shape[1],
        indices.shape[2],
        indices.shape[3],
        key.shape[3],
    )

    value = value.gather(
        dim=2,
        index=expanded_indices,
    ).view(
        value.shape[0],
        value.shape[1],
        indices.shape[2],
        indices.shape[3],
        value.shape[3],
    )

    del indices, expanded_indices

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
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
    )

    attn_output = unstack_attn(attn_output, batch_size, origional_seq_len)

    return attn_output, None


gpu_ok = False
if torch.cuda.is_available():
    device_cap = torch.cuda.get_device_capability()
    if device_cap in ((7, 0), (8, 0), (9, 0)):
        gpu_ok = True


if gpu_ok:
    print("Using compiled dynamic attention sinks attention implementation.")
    dynamic_attention_sinks_attention_forward = torch.compile(
        das_attention_parallel_forward
    )
else:
    print("Using eager dynamic attention sinks attention implementation.")
    dynamic_attention_sinks_attention_forward = das_attention_parallel_forward
