from typing import Optional, Self

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention_mapping import AttentionMapping, T


class LinearConvAttentionMapping(AttentionMapping):
    conv1d: Optional[nn.Conv1d] = None
    num_full_layers: Optional[int] = None
    num_full_heads: Optional[int] = None

    def __init__(
        self,
        path: Optional[str] = None,
        old_path: bool = False,
        kernel_size: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.dtype = dtype
        self.device = device
        if path is not None:
            parameters = torch.load(path, weights_only=False, map_location=device)
            if old_path:
                assert "kernel" in parameters and "kernel_size" in parameters
                kernel = parameters["kernel"].to(device, dtype)
                self.kernel_size: int = parameters["kernel_size"]
                weight = kernel.permute(3, 2, 1, 0, 4).reshape(
                    kernel.shape[3] * kernel.shape[2],
                    kernel.shape[1] * kernel.shape[0],
                    kernel.shape[4],
                )
                self.conv1d = torch.nn.Conv1d(
                    in_channels=weight.shape[1],
                    out_channels=weight.shape[0],
                    kernel_size=weight.shape[2],
                    stride=1,
                    padding="same",
                    padding_mode="replicate",
                    bias=False,
                    device=self.device,
                    dtype=self.dtype,
                )
                self.conv1d.weight.requires_grad = False
                self.conv1d.weight.copy_(weight)
                self.num_full_layers = kernel.shape[3]
                self.num_full_heads = kernel.shape[2]
            else:
                assert (
                    "conv1d" in parameters
                    and "num_full_layers" in parameters
                    and "num_full_heads"
                )
                self.conv1d = parameters["conv1d"]
                self.num_full_layers = parameters["num_full_layers"]
                self.num_full_heads = parameters["num_full_heads"]
                assert self.conv1d is not None
                self.kernel_size = self.conv1d.weight.shape[-1]
        else:
            assert kernel_size is not None and kernel_size % 2 == 1
            self.kernel_size = kernel_size

    def stack_matrix(self, a: Tensor):
        pad = self.kernel_size // 2
        front_padding = a[..., :1].expand(-1, -1, -1, pad)
        end_padding = a[..., -1:].expand(-1, -1, -1, pad)
        a = torch.cat((front_padding, a, end_padding), dim=-1)
        a = torch.stack(
            [
                a[..., i : a.shape[-1] - self.kernel_size + i + 1]
                for i in range(self.kernel_size)
            ],
            dim=-1,
        )
        a = F.pad(
            a.transpose(0, 3).reshape(a.shape[3] * a.shape[1], -1),
            pad=(0, 1),
            value=1,
        )
        return a.contiguous()

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
            x = self.stack_matrix(x).to(self.device, self.dtype)
            y = y.transpose(0, 3).reshape(-1, n).to(self.device, self.dtype)
            a += x.T @ x / (x.shape[0] ** 2)
            b += x.T @ y / (x.shape[0] ** 2)

        if weight_decay != 0:
            a += weight_decay * torch.eye(m + 1, dtype=self.dtype, device=self.device)

        affine_weight: Tensor = torch.linalg.solve(a, b)

        weight = (
            affine_weight[:-1]
            .reshape(
                num_draft_heads,
                num_draft_layers,
                self.kernel_size,
                num_full_heads,
                num_full_layers,
            )
            .permute(4, 3, 1, 0, 2)
            .reshape(n, -1, self.kernel_size)
        )

        self.conv1d = torch.nn.Conv1d(
            in_channels=weight.shape[1],
            out_channels=weight.shape[0],
            kernel_size=weight.shape[2],
            stride=1,
            padding="same",
            padding_mode="replicate",
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )
        self.conv1d.weight.requires_grad = False
        self.conv1d.weight.copy_(weight)

        self.num_full_layers = num_full_layers
        self.num_full_heads = num_full_heads

        return self

    def stack(self, a: Tensor) -> Tensor:
        num_layers, batch_size, num_heads, seq_len = a.shape
        return a.transpose(0, 1).reshape(batch_size, num_layers * num_heads, seq_len)

    def unstack(self, a: Tensor) -> Tensor:
        assert self.num_full_layers is not None and self.num_full_heads is not None
        batch_size, _, seq_len = a.shape
        return a.view(
            batch_size,
            self.num_full_layers,
            self.num_full_heads,
            seq_len,
        ).transpose(0, 1)

    def map_single(self, a: Tensor) -> Tensor:
        assert self.conv1d is not None

        if a.device != self.device or a.dtype != self.dtype:
            a = a.to(self.device, self.dtype)

        return self.unstack(self.conv1d(self.stack(a)))

    def __call__(self, draft_reduced_attentions: T) -> T:
        if isinstance(draft_reduced_attentions, list):
            return [self.map_single(a) for a in draft_reduced_attentions]
        else:
            return self.map_single(draft_reduced_attentions)

    def save(self, path: str) -> None:
        assert self.conv1d is not None
        torch.save(
            {
                "conv1d": self.conv1d,
                "num_full_layers": self.num_full_layers,
                "num_full_heads": self.num_full_heads,
            },
            path,
        )
