import torch
from minference import vertical_slash_sparse_attention  # type: ignore
from torch import Tensor
from transformers.models.llama.modeling_llama import (  # type: ignore
    LlamaAttention,
    repeat_kv,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention  # type: ignore


def das_minference_attention_forward(
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


def compress_states(states: Tensor, v_idx: Tensor, window_size: int) -> Tensor:
    sink_indices = v_idx[..., None].expand(-1, -1, -1, states.shape[-1])
    sink_states = states.gather(dim=2, index=sink_indices.long())
    window_states = states[..., -window_size:, :]
    return torch.cat((sink_states, window_states), dim=2)
