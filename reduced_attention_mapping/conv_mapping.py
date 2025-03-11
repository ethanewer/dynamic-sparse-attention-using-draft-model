from typing import Self, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.notebook import trange  # type: ignore

from .attention_mapping import AttentionMapping, T


class ConvAttentionMapping(AttentionMapping):
    model: Optional[nn.Sequential] = None
    num_full_layers: Optional[int] = None
    num_full_heads: Optional[int] = None
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device | str] = None,
    

    def __init__(self, path: Optional[str] = None) -> None:
        if path is not None:
            self.model = torch.load(path, weights_only=False)
            self.model.eval()

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
        assert self.model is not None and self.dtype is not None and self.device is not None

        if a.dtype != self.dtype:
            a = a.to(dtype=self.dtype)

        if a.device != self.device:
            a = a.to(device=self.device)

        a /= a.sum(dim=-1)[..., None]
        
        return self.unstack(self.model(self.stack(a)))
        

    def __call__(self, draft_reduced_attentions: T) -> T:
        if isinstance(draft_reduced_attentions, list):
            return [self.map_single(a) for a in draft_reduced_attentions]
        else:
            return self.map_single(draft_reduced_attentions)

    def save(self, path: str) -> None:
        assert self.model is not None
        torch.save(self.model, path)


class LinearConvMapping(ConvAttentionMapping):
    def fit(
        self,
        draft_reduced_attentions: list[Tensor],
        full_reduced_attentions: list[Tensor],
        kernel_size: int = 5,
        num_iters: int = 10,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> Self:
        num_draft_layers = draft_reduced_attentions[0].shape[0]
        num_draft_heads = draft_reduced_attentions[0].shape[2]
        num_full_layers = full_reduced_attentions[0].shape[0]
        num_full_heads = full_reduced_attentions[0].shape[2]

        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels=num_draft_layers * num_draft_heads,
                out_channels=num_full_layers * num_full_heads,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                padding_mode="replicate",
                bias=None,
                device=device,
                dtype=dtype,
            ),
            nn.Softmax(dim=-1),
        )

        self.num_full_heads = num_full_heads
        self.num_full_layers = num_full_layers
        self.dtype = dtype
        self.device = device

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iters)

        progress_bar = trange(num_iters, desc="[loss: NaN]")
        for _ in progress_bar:
            losses = []
            for i in np.random.permutation(len(draft_reduced_attentions)):
                x = draft_reduced_attentions[i].to(device, dtype)
                y = full_reduced_attentions[i].to(device, dtype)
                y /= y.sum(dim=-1)[..., None]
                y_hat = self(x)
                optimizer.zero_grad()
                loss = F.kl_div(y_hat.log(), y, reduction="batchmean")
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            scheduler.step()
            progress_bar.set_description(f"[loss: {sum(losses) / len(losses):.4f}]")

        return self
    

class NonlinearConvMapping(ConvAttentionMapping):
    def fit(
        self,
        draft_reduced_attentions: list[Tensor],
        full_reduced_attentions: list[Tensor],
        kernel_size: int = 5,
        num_iters: int = 10,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> Self:
        num_draft_layers = draft_reduced_attentions[0].shape[0]
        num_draft_heads = draft_reduced_attentions[0].shape[2]
        num_full_layers = full_reduced_attentions[0].shape[0]
        num_full_heads = full_reduced_attentions[0].shape[2]

        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels=num_draft_layers * num_draft_heads,
                out_channels=2 * num_full_layers * num_full_heads,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                padding_mode="replicate",
                bias=None,
                device=device,
                dtype=dtype,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=2 * num_full_layers * num_full_heads,
                out_channels=num_full_layers * num_full_heads,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                padding_mode="replicate",
                bias=None,
                device=device,
                dtype=dtype,
            ),
            nn.Softmax(dim=-1),
        )

        self.num_full_heads = num_full_heads
        self.num_full_layers = num_full_layers
        self.dtype = dtype
        self.device = device

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iters)

        progress_bar = trange(num_iters, desc="[loss: NaN]")
        for _ in progress_bar:
            losses = []
            for i in np.random.permutation(len(draft_reduced_attentions)):
                x = draft_reduced_attentions[i].to(device, dtype)
                y = full_reduced_attentions[i].to(device, dtype)
                y /= y.sum(dim=-1)[..., None]
                y_hat = self(x)
                optimizer.zero_grad()
                loss = F.kl_div(y_hat.log(), y, reduction="batchmean")
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            scheduler.step()
            progress_bar.set_description(f"[loss: {sum(losses) / len(losses):.4f}]")

        return self