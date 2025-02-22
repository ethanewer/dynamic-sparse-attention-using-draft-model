import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SnapKVCluster:
    def __init__(
        self,
        window_size=32,
        max_capacity_prompt=2048,
        kernel_size=5,
        pooling="avgpool",
    ) -> None:
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt > self.window_size
        self.kernel_size = kernel_size
        self.pooling = pooling

    @staticmethod
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

    def update_kv(
        self,
        key_states: Tensor,
        query_states: Tensor,
        value_states: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]

        # average queries for group query attention
        if query_states.shape[1] > key_states.shape[1]:
            queries_per_key_value = query_states.shape[1] // key_states.shape[1]
            query_states = self.pool_queries_for_gqa(
                query_states,
                queries_per_key_value,
            )
            assert query_states.shape[1] == key_states.shape[1]

        _, _, seq_len, head_dim = query_states.shape
        if seq_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attention_weights = torch.matmul(
                query_states[..., -self.window_size :, :],
                key_states.transpose(2, 3),
            ) / math.sqrt(head_dim)
            mask = torch.full(
                (self.window_size, self.window_size),
                torch.finfo(attention_weights.dtype).min,
                device=attention_weights.device,
            )
            mask_cond = torch.arange(mask.size(-1), device=attention_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attention_weights.device)
            attention_mask = mask[None, None, :, :]

            attention_weights[:, :, -self.window_size :, -self.window_size :] += (
                attention_mask
            )

            attention_weights = nn.functional.softmax(
                attention_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attention_weights_sum = attention_weights[
                :, :, -self.window_size :, : -self.window_size
            ].sum(dim=-2)
            if self.pooling == "avgpool":
                attention_cache = F.avg_pool1d(
                    attention_weights_sum,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    stride=1,
                )
            elif self.pooling == "maxpool":
                attention_cache = F.max_pool1d(
                    attention_weights_sum,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    stride=1,
                )
            else:
                raise ValueError("Pooling method not supported")

            indices = attention_cache.topk(
                self.max_capacity_prompt - self.window_size,
                dim=-1,
            ).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            key_past_compress = key_states[:, :, : -self.window_size, :].gather(
                dim=2,
                index=indices,
            )
            value_past_compress = value_states[:, :, : -self.window_size, :].gather(
                dim=2,
                index=indices,
            )
            key_window = key_states[:, :, -self.window_size :, :]
            value_window = value_states[:, :, -self.window_size :, :]
            key_states = torch.cat([key_past_compress, key_window], dim=2)
            value_states = torch.cat([value_past_compress, value_window], dim=2)

            return key_states, value_states


def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, "window_size"):
            self.config.window_size = 32
        if not hasattr(self.config, "max_capacity_prompt"):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, "kernel_size"):
            self.config.kernel_size = 5
        if not hasattr(self.config, "pooling"):
            self.config.pooling = "avgpool"

    self.kv_cluster = SnapKVCluster(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        kernel_size=self.config.kernel_size,
        pooling=self.config.pooling,
    )
