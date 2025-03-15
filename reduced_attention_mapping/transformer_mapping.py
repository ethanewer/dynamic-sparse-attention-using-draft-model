from typing import Optional, Self

import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor, nn
from tqdm.notebook import trange  # type: ignore

from .attention_mapping import AttentionMapping, T


class SingleHeadAttention(nn.Module):
    def __init__(
        self,
        num_draft_layers: int,
        num_draft_heads: int,
        num_full_layers: int,
        num_full_heads: int,
        seq_len_as_hidden_dim: bool,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.num_draft_layers = num_draft_layers
        self.num_draft_heads = num_draft_heads
        self.num_full_layers = num_full_layers
        self.num_full_heads = num_full_heads
        self.seq_len_as_hidden_dim = seq_len_as_hidden_dim

        input_size = num_draft_layers * num_draft_heads
        output_size = num_full_layers * num_full_heads

        if self.seq_len_as_hidden_dim:
            self.q_proj = nn.Linear(input_size, output_size, device=device, dtype=dtype)
            self.k_proj = nn.Linear(input_size, input_size, device=device, dtype=dtype)
            self.v_proj = nn.Linear(input_size, input_size, device=device, dtype=dtype)
        else:
            self.q_proj = nn.Linear(input_size, input_size, device=device, dtype=dtype)
            self.k_proj = nn.Linear(input_size, input_size, device=device, dtype=dtype)
            self.v_proj = nn.Linear(input_size, output_size, device=device, dtype=dtype)

    def forward(self, reduced_attentions: Tensor) -> Tensor:
        _, batch_size, _, seq_len = reduced_attentions.shape

        reduced_attentions /= reduced_attentions.sum(dim=-1)[..., None]

        reduced_attentions = rearrange(reduced_attentions, "l b h t -> b t (l h)")

        query = self.q_proj(reduced_attentions)
        key = self.k_proj(reduced_attentions)
        value = self.v_proj(reduced_attentions)

        if self.seq_len_as_hidden_dim:
            attn_weights = einsum(query, key, "b t lh2, b t lh1 -> b lh2 lh1")
            attn_weights = F.softmax(
                attn_weights,
                dim=-1,
                dtype=torch.float32,
            ).to(attn_weights.dtype)
            attn_output = einsum(attn_weights, value, "b lh2 lh1, b t lh1 -> b t lh2")
        else:
            attn_weights = einsum(query, key, "b t1 lh1, b t2 lh1 -> b t1 t2")
            attn_weights = F.softmax(
                attn_weights,
                dim=-1,
                dtype=torch.float32,
            ).to(attn_weights.dtype)
            attn_output = einsum(attn_weights, value, "b t1 t2, b t2 lh2 -> b t1 lh2")

        attn_output = rearrange(
            attn_output,
            "b t (l2 h2) -> l2 b h2 t",
            b=batch_size,
            t=seq_len,
            l2=self.num_full_layers,
            h2=self.num_full_heads,
        )

        return attn_output


class AttentionAttentionMapping(AttentionMapping):
    model: Optional[nn.Module] = None

    def __init__(
        self,
        path: Optional[str] = None,
        seq_len_as_hidden_dim: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.seq_len_as_hidden_dim = seq_len_as_hidden_dim
        self.dtype = dtype
        self.device = device
        if path is not None:
            parameters = torch.load(path, weights_only=False, map_location=device)
            self.model = parameters["model"].to(device, dtype)

    def fit(
        self,
        draft_reduced_attentions: list[Tensor],
        full_reduced_attentions: list[Tensor],
        test_draft_reduced_attentions: Optional[list[Tensor]] = None,
        test_full_reduced_attentions: Optional[list[Tensor]] = None,
        num_iters: int = 10,
        lr: float = 5e-4,
    ) -> Self:
        self.model = SingleHeadAttention(
            num_draft_layers=draft_reduced_attentions[0].shape[0],
            num_draft_heads=draft_reduced_attentions[0].shape[2],
            num_full_layers=full_reduced_attentions[0].shape[0],
            num_full_heads=full_reduced_attentions[0].shape[2],
            seq_len_as_hidden_dim=self.seq_len_as_hidden_dim,
            dtype=self.dtype,
            device=self.device,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iters)
        min_test_loss = float("inf")

        progress_bar = trange(num_iters, desc="[]")
        for _ in progress_bar:
            train_losses = []
            for i in np.random.permutation(len(draft_reduced_attentions)):
                x = draft_reduced_attentions[i].to(self.device, self.dtype)
                y = full_reduced_attentions[i].to(self.device, self.dtype)
                y /= y.sum(dim=-1)[..., None]
                optimizer.zero_grad()
                loss = F.kl_div(self(x).log(), y, reduction="batchmean")
                if torch.isfinite(loss):
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

            scheduler.step()
            train_loss = sum(train_losses) / len(train_losses)

            if (
                test_draft_reduced_attentions is not None
                and test_full_reduced_attentions is not None
            ):
                test_losses = []
                for i in range(len(test_draft_reduced_attentions)):
                    x = test_draft_reduced_attentions[i].to(self.device, self.dtype)
                    y = test_full_reduced_attentions[i].to(self.device, self.dtype)
                    y /= y.sum(dim=-1)[..., None]
                    log_y_pred = self.model(x).log_softmax(dim=-1)
                    with torch.no_grad():
                        loss = F.kl_div(log_y_pred, y, reduction="batchmean")

                    test_losses.append(loss.item())

                test_loss = sum(test_losses) / len(test_losses)
                min_test_loss = min(min_test_loss, test_loss)
                progress_bar.set_description(
                    f"[train loss: {train_loss:.4f}, test loss: {test_loss:.4f}, min test loss: {min_test_loss:.4f}]"
                )
            else:
                progress_bar.set_description(f"[train loss: {train_loss:.4f}]")

        for p in self.model.parameters():
            p.requires_grad = False

        return self

    def __call__(self, draft_reduced_attentions: T) -> T:
        assert self.model is not None
        if isinstance(draft_reduced_attentions, list):
            return [
                self.model(a.to(self.device, self.dtype))
                for a in draft_reduced_attentions
            ]
        else:
            return self.model(draft_reduced_attentions.to(self.device, self.dtype))

    def save(self, path: str) -> None:
        assert self.model is not None
        torch.save({"model": self.model}, path)
