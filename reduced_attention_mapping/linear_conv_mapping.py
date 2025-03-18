from typing import Optional, Self

import torch
import torch.nn.functional as F
from torch import Tensor

from .attention_mapping import AttentionMapping, T


class LinearConvAttentionMapping(AttentionMapping):
    kernel: Optional[Tensor] = None

    def __init__(
        self,
        path: Optional[str] = None,
        kernel_size: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.dtype = dtype
        self.device = device
        if path is not None:
            parameters = torch.load(path, weights_only=False, map_location=device)
            assert "kernel" in parameters and "kernel_size" in parameters
            self.kernel = parameters["kernel"].to(device, dtype)
            self.kernel_size = parameters["kernel_size"]
        else:
            assert kernel_size is not None and kernel_size % 2 == 1
            self.kernel_size = kernel_size

    def stack(self, a: Tensor):
        pad = self.kernel_size // 2
        front_padding = a[..., :1].expand(-1, -1, -1, pad)
        end_padding = a[..., -1:].expand(-1, -1, -1, pad)
        a = torch.cat((front_padding, a, end_padding), dim=-1)
        return torch.stack(
            [
                a[..., i : a.shape[-1] - self.kernel_size + i + 1]
                for i in range(self.kernel_size)
            ],
            dim=-1,
        )

    def fit(
        self,
        draft_reduced_attentions: list[Tensor],
        full_reduced_attentions: list[Tensor],
        weight_decay: float = 0.0,
    ) -> Self:
        num_draft_layers = draft_reduced_attentions[0].shape[0]
        num_draft_heads = draft_reduced_attentions[0].shape[2]
        num_full_layers = full_reduced_attentions[0].shape[0]
        num_full_heads = full_reduced_attentions[0].shape[2]

        m = num_draft_layers * num_draft_heads * self.kernel_size
        n = num_full_layers * num_full_heads

        a = torch.zeros(m + 1, m + 1, dtype=self.dtype, device=self.device)
        b = torch.zeros(m + 1, n, dtype=self.dtype, device=self.device)

        for x, y in zip(draft_reduced_attentions, full_reduced_attentions):
            x = F.pad(
                self.stack(x).transpose(0, 3).reshape(-1, m),
                pad=(0, 1),
                value=1,
            ).to(self.device, self.dtype)
            y = y.transpose(0, 3).reshape(-1, n).to(self.device, self.dtype)
            a += x.T @ x / (x.shape[0] ** 2)
            b += x.T @ y / (x.shape[0] ** 2)

        if weight_decay != 0:
            a += weight_decay * torch.eye(m + 1, dtype=self.dtype, device=self.device)

        affine_kernel: Tensor = torch.linalg.solve(a, b)

        self.kernel = (
            affine_kernel[:-1]
            .reshape(
                num_draft_heads,
                num_draft_layers,
                self.kernel_size,
                num_full_heads,
                num_full_layers,
            )
            .permute(0, 1, 3, 4, 2)
        )

        return self

    def map_single(self, a: Tensor) -> Tensor:
        assert self.kernel is not None

        if a.device != self.device or a.dtype != self.dtype:
            a = a.to(self.device, self.dtype)

        return torch.einsum("lbhtk,hlHLk->LbHt", self.stack(a), self.kernel)

    def __call__(self, draft_reduced_attentions: T) -> T:
        if isinstance(draft_reduced_attentions, list):
            return [self.map_single(a) for a in draft_reduced_attentions]
        else:
            return self.map_single(draft_reduced_attentions)

    def save(self, path: str) -> None:
        assert self.kernel is not None
        torch.save({"kernel": self.kernel, "kernel_size": self.kernel_size}, path)
