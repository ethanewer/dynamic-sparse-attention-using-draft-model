from typing import Literal, Optional, Self

import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum
from torch import Tensor, nn
from tqdm.notebook import tqdm, trange  # type: ignore

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

        if a.device != self.device or a.dtype != self.dtype:
            a = a.to(self.device, self.dtype)

        return einsum(a, self.w, "l1 b h1 t, l1 h1 l2 h2 -> l2 b h2 t")

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
        num_iters: int = 32,
        lr: float = 1e-3,
        lr_decay: float = 0.1,
    ) -> Self:
        num_draft_layers = draft_reduced_attentions[0].shape[0]
        num_draft_heads = draft_reduced_attentions[0].shape[2]
        num_full_layers = full_reduced_attentions[0].shape[0]
        num_full_heads = full_reduced_attentions[0].shape[2]

        w_shape = (
            num_draft_layers,
            num_draft_heads,
            num_full_layers,
            num_full_heads,
        )
        unnormalized_w = torch.zeros(
            num_draft_heads * num_draft_layers,
            num_full_heads * num_full_layers,
            requires_grad=True,
            dtype=self.dtype,
            device=self.device,
        )

        optimizer = torch.optim.Adam([unnormalized_w], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            num_iters,
            lr * lr_decay,
        )
        progress_bar = trange(num_iters, desc="[]")
        for _ in progress_bar:
            losses = []
            permutation: list[int] = np.random.permutation(
                len(draft_reduced_attentions)
            ).tolist()
            for i in permutation:
                x = draft_reduced_attentions[i].to(self.device, self.dtype)
                y = full_reduced_attentions[i].to(self.device, self.dtype)
                self.w = unnormalized_w.softmax(dim=0).view(w_shape)
                loss = F.kl_div(self(x).log(), y, reduction="batchmean")
                if torch.isfinite(loss):
                    optimizer.zero_grad()
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
            (num_draft_layers, num_draft_heads, num_full_layers, num_full_heads)
        )

        for i in trange(num_full_layers):
            for j in range(num_full_heads):
                scores = np.zeros((num_draft_layers, num_draft_heads))
                for x, y in zip(draft_topks, full_topks):
                    scores += np.isin(x, y[i, j]).sum(axis=2)

                k_max, l_max = np.unravel_index(scores.argmax(), scores.shape)
                w[k_max, l_max, i, j] = 1

        self.w = torch.tensor(w, dtype=self.dtype, device=self.device)

        return self


class LinearConvAttentionMapping(AttentionMapping):
    conv1d: Optional[nn.Conv1d] = None
    num_full_layers: Optional[int] = None
    num_full_heads: Optional[int] = None

    def __init__(
        self,
        path: Optional[str] = None,
        kernel_size: Optional[int] = None,
        normalized_reduced_attentions: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.dtype = dtype
        self.device = device
        self.normalized_reduced_attentions = normalized_reduced_attentions
        if path is not None:
            parameters = torch.load(path, weights_only=False, map_location=device)
            assert (
                "conv1d" in parameters
                and "num_full_layers" in parameters
                and "num_full_heads"
            )
            self.conv1d = parameters["conv1d"].to(self.device, self.dtype)
            self.num_full_layers = parameters["num_full_layers"]
            self.num_full_heads = parameters["num_full_heads"]
            assert self.conv1d is not None
            self.kernel_size = self.conv1d.weight.shape[-1]
        else:
            assert kernel_size is not None and kernel_size % 2 == 1
            self.kernel_size = kernel_size

    def scale(self, a: Tensor, eps: float = 1e-6) -> Tensor:
        a += (eps - a.min(dim=-1).values[..., None]).relu()
        a /= a.sum(dim=-1)[..., None]
        return a

    def stack_matrix(self, a: Tensor) -> Tensor:
        pad = self.kernel_size // 2
        a = F.pad(a, (pad, pad, 0, 0), mode="replicate")
        a = a.unfold(dimension=-1, size=self.kernel_size, step=1)
        a = a.transpose(0, 3).reshape(a.shape[3] * a.shape[1], -1)
        a = F.pad(a, pad=(0, 1), value=torch.e)
        return a.contiguous()

    def fit_lstsq(
        self,
        draft_reduced_attentions: list[Tensor],
        full_reduced_attentions: list[Tensor],
        weight_decay: float = 0.0,
    ) -> None:
        num_draft_layers = draft_reduced_attentions[0].shape[0]
        num_draft_heads = draft_reduced_attentions[0].shape[2]
        num_full_layers = full_reduced_attentions[0].shape[0]
        num_full_heads = full_reduced_attentions[0].shape[2]

        m = num_draft_layers * num_draft_heads * self.kernel_size
        n = num_full_layers * num_full_heads

        a = torch.zeros(m + 1, m + 1, dtype=self.dtype, device=self.device)
        b = torch.zeros(m + 1, n, dtype=self.dtype, device=self.device)

        for x, y in tqdm(
            list(zip(draft_reduced_attentions, full_reduced_attentions)),
            desc="building matrix",
        ):
            x = x.to(self.device, self.dtype)
            y = y.to(self.device, self.dtype)
            if self.normalized_reduced_attentions:
                x = self.scale(x)
                y = self.scale(y)

            x = self.stack_matrix(x)
            y = y.transpose(0, 3).reshape(-1, n)

            a += x.T @ x / (x.shape[0] ** 2)
            b += x.T @ y / (x.shape[0] ** 2)

        if weight_decay != 0:
            a += weight_decay * torch.eye(m + 1, dtype=self.dtype, device=self.device)

        affine_weight: Tensor = torch.linalg.lstsq(a, b)[0]

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

    def kl_div_loss(
        self,
        output: Tensor,
        target: Tensor,
    ) -> Tensor:
        if self.normalized_reduced_attentions:
            output = self.scale(output).log()
            target = self.scale(target)
        else:
            output = output.log_softmax(dim=-1)
            target = target.softmax(dim=-1)

        return F.kl_div(output, target, reduction="batchmean")

    def fit_kl_div_adam(
        self,
        draft_reduced_attentions: list[Tensor],
        full_reduced_attentions: list[Tensor],
        num_iters: int,
        lr: float,
        lr_decay: float,
        weight_decay: float,
    ) -> None:
        self.num_draft_layers = draft_reduced_attentions[0].shape[0]
        self.num_draft_heads = draft_reduced_attentions[0].shape[2]
        self.num_full_layers = full_reduced_attentions[0].shape[0]
        self.num_full_heads = full_reduced_attentions[0].shape[2]

        self.conv1d = torch.nn.Conv1d(
            in_channels=self.num_draft_layers * self.num_draft_heads,
            out_channels=self.num_full_layers * self.num_full_heads,
            kernel_size=self.kernel_size,
            stride=1,
            padding="same",
            padding_mode="replicate",
            bias=True,
            device=self.device,
            dtype=self.dtype,
        )
        optimizer = torch.optim.AdamW(
            self.conv1d.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            num_iters,
            lr * lr_decay,
        )
        progress_bar = trange(num_iters, desc="[]")
        for _ in progress_bar:
            losses = []
            permutation: list[int] = np.random.permutation(
                len(draft_reduced_attentions)
            ).tolist()
            for i in permutation:
                x = draft_reduced_attentions[i].to(self.device, self.dtype)
                y = full_reduced_attentions[i].to(self.device, self.dtype)
                loss = self.kl_div_loss(self(x), y)
                if torch.isfinite(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

            scheduler.step()
            progress_bar.set_description(f"[loss: {sum(losses) / len(losses):.4f}]")

        for p in self.conv1d.parameters():
            p.requires_grad = False

    def fit(
        self,
        draft_reduced_attentions: list[Tensor],
        full_reduced_attentions: list[Tensor],
        objective: Literal["mse", "kl_div"] = "kl_div",
        num_iters: int = 32,
        lr: float = 1e-3,
        lr_decay: float = 0.1,
        weight_decay: float = 0.0,
    ) -> Self:
        if objective == "mse":
            self.fit_lstsq(
                draft_reduced_attentions,
                full_reduced_attentions,
                weight_decay=weight_decay,
            )
        elif objective == "kl_div":
            self.fit_kl_div_adam(
                draft_reduced_attentions,
                full_reduced_attentions,
                num_iters=num_iters,
                lr=lr,
                lr_decay=lr_decay,
                weight_decay=weight_decay,
            )
        else:
            raise NotImplementedError

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

        if self.normalized_reduced_attentions:
            a = self.scale(a)

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
