from typing import Optional, Self

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.notebook import trange  # type: ignore

from .attention_mapping import AttentionMapping, T


class BaseLinearAttentionMapping(AttentionMapping):
    w: Optional[Tensor] = None

    def __init__(
        self,
        path: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.dtype = dtype
        self.device = device
        if path is not None:
            parameters = torch.load(path, weights_only=False, map_location=device)
            assert "w" in parameters
            self.w = parameters["w"].to(device, dtype)

    def map_single(self, a: Tensor) -> Tensor:
        assert self.w is not None

        if a.dtype != self.dtype:
            a = a.to(dtype=self.dtype)

        if a.device != self.device:
            a = a.to(device=self.device)

        return torch.einsum("lbht,hlHL->LbHt", a, self.w)

    def __call__(self, draft_reduced_attentions: T) -> T:
        if isinstance(draft_reduced_attentions, list):
            return [self.map_single(a) for a in draft_reduced_attentions]
        else:
            return self.map_single(draft_reduced_attentions)

    def save(self, path: str) -> None:
        assert self.w is not None
        torch.save({"w": self.w}, path)


class LinearAttentionMapping(BaseLinearAttentionMapping):
    def fit(
        self,
        draft_reduced_attentions: list[Tensor],
        full_reduced_attentions: list[Tensor],
        num_iters: int = 10,
        lr: float = 1e-3,
    ) -> Self:
        num_draft_layers = draft_reduced_attentions[0].shape[0]
        num_draft_heads = draft_reduced_attentions[0].shape[2]
        num_full_layers = full_reduced_attentions[0].shape[0]
        num_full_heads = full_reduced_attentions[0].shape[2]

        w_shape = (
            num_draft_heads,
            num_draft_layers,
            num_full_heads,
            num_full_layers,
        )
        unnormalized_w = torch.zeros(
            num_draft_heads * num_draft_layers,
            num_full_heads * num_full_layers,
            requires_grad=True,
            dtype=self.dtype,
            device=self.device,
        )

        optimizer = torch.optim.Adam([unnormalized_w], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iters)

        progress_bar = trange(num_iters, desc="[]")
        for _ in progress_bar:
            losses = []
            for i in np.random.permutation(len(draft_reduced_attentions)):
                x = draft_reduced_attentions[i].to(self.device, self.dtype)
                y = full_reduced_attentions[i].to(self.device, self.dtype)
                optimizer.zero_grad()
                self.w = unnormalized_w.softmax(dim=0).view(w_shape)
                loss = F.kl_div(self(x).log(), y, reduction="batchmean")
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            scheduler.step()
            progress_bar.set_description(f"[loss: {sum(losses) / len(losses):.4f}]")

        self.w = unnormalized_w.softmax(dim=0).view(w_shape).detach()
        return self


class AverageAttentionMapping(BaseLinearAttentionMapping):
    def fit(
        self,
        draft_reduced_attentions: list[Tensor],
        full_reduced_attentions: list[Tensor],
    ) -> Self:
        num_draft_layers = draft_reduced_attentions[0].shape[0]
        num_draft_heads = draft_reduced_attentions[0].shape[2]
        num_full_layers = full_reduced_attentions[0].shape[0]
        num_full_heads = full_reduced_attentions[0].shape[2]

        self.w = torch.full(
            (num_draft_heads, num_draft_layers, num_full_heads, num_full_layers),
            fill_value=1 / (num_draft_heads * num_draft_layers),
            dtype=self.dtype,
            device=self.device,
        )

        return self


class GreedyAttentionMapping(BaseLinearAttentionMapping):
    @staticmethod
    def seperate_batch(reduced_attentions: list[Tensor]) -> list[Tensor]:
        seperated_reduced_attentions = []
        for a in reduced_attentions:
            for i in range(a.shape[1]):
                seperated_reduced_attentions.append(a[:, i])

        return seperated_reduced_attentions

    def fit(
        self,
        draft_reduced_attentions: list[Tensor],
        full_reduced_attentions: list[Tensor],
        r: float = 0.125,
    ) -> Self:
        num_draft_layers = draft_reduced_attentions[0].shape[0]
        num_draft_heads = draft_reduced_attentions[0].shape[2]
        num_full_layers = full_reduced_attentions[0].shape[0]
        num_full_heads = full_reduced_attentions[0].shape[2]

        draft_topks = [
            a.topk(int(r * a.shape[-1])).indices.numpy()
            for a in self.seperate_batch(draft_reduced_attentions)
        ]
        full_topks = [
            a.topk(int(r * a.shape[-1])).indices.numpy()
            for a in self.seperate_batch(full_reduced_attentions)
        ]

        w = np.zeros(
            (num_draft_heads, num_draft_layers, num_full_heads, num_full_layers)
        )

        for i in trange(num_full_heads):
            for j in range(num_full_layers):
                scores = np.zeros((num_draft_heads, num_draft_layers))
                for x, y in zip(draft_topks, full_topks):
                    scores += np.isin(x, y[j, i]).sum(axis=2).T

                k_max, l_max = np.unravel_index(scores.argmax(), scores.shape)
                w[k_max, l_max, i, j] = 1

        self.w = torch.tensor(w, dtype=self.dtype, device=self.device)

        return self
