from typing import Any, Optional

import mlx.core as mx
from mlx import nn
from mlx_lm.models.cache import QuantizedKVCache
from mlx_lm.models.qwen2 import Attention


def repeat_kv(hidden_states: mx.array, n_rep: int) -> mx.array:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = mx.repeat(hidden_states[:, :, None, :, :], n_rep, axis=2)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_scaled_dot_product_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Optional[mx.array],
) -> tuple[mx.array, mx.array]:
    num_key_value_groups = queries.shape[1] // keys.shape[1]
    key_states = repeat_kv(keys, num_key_value_groups)
    value_states = repeat_kv(values, num_key_value_groups)

    attn_weights = (queries @ key_states.transpose(0, 1, 3, 2)) * scale
    if mask is not None:
        causal_mask = mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = mx.softmax(attn_weights, axis=-1)
    attn_output = (attn_weights @ value_states).transpose(0, 2, 1, 3)
    return attn_output, attn_weights


class AttentionForOutputtingScores(nn.Module):
    def __init__(self, self_attn: Attention, attn_log: list):
        super().__init__()
        self.n_heads = self_attn.n_heads
        self.n_kv_heads = self_attn.n_kv_heads
        self.head_dim = self_attn.head_dim
        self.scale = self_attn.scale
        self.self_attn = self_attn
        self.attn_log = attn_log

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if isinstance(cache, QuantizedKVCache):
            raise NotImplementedError

        B, L, D = x.shape

        queries, keys, values = (
            self.self_attn.q_proj(x),
            self.self_attn.k_proj(x),
            self.self_attn.v_proj(x),
        )

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.self_attn.rope(queries, offset=cache.offset)
            keys = self.self_attn.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.self_attn.rope(queries)
            keys = self.self_attn.rope(keys)

        output, attns = eager_scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        self.attn_log.append(attns)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.self_attn.o_proj(output)


def update_qwen2_model_to_output_attns(model) -> list:
    attn_log: list = []
    for i in range(len(model.model.layers)):
        model.model.layers[i].self_attn = AttentionForOutputtingScores(
            self_attn=model.model.layers[i].self_attn,
            attn_log=attn_log,
        )

    return attn_log


def reset_qwen2_model(model) -> None:
    for i in range(len(model.model.layers)):
        model.model.layers[i].self_attn = model.model.layers[i].self_attn.self_attn
