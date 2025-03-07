from typing import Self

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.notebook import trange  # type: ignore

from .attention_mapping import AttentionMapping


class MSEAttentionMapping(AttentionMapping):
    def fit(
        self,
        draft_reduced_attentions: list[Tensor],
        full_reduced_attentions: list[Tensor],
        n_iters: int = 10,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> Self:
        num_draft_layers = draft_reduced_attentions[0].shape[0]
        num_draft_heads = draft_reduced_attentions[0].shape[2]
        num_full_layers = full_reduced_attentions[0].shape[0]
        num_full_heads = full_reduced_attentions[0].shape[2]

        self.w_shape = (
            num_draft_heads,
            num_draft_layers,
            num_full_heads,
            num_full_layers,
        )
        self.w = torch.zeros(
            num_draft_heads * num_draft_layers,
            num_full_heads * num_full_layers,
            requires_grad=True,
            device=device,
        )

        optimizer = torch.optim.Adam([self.w], lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iters)

        progress_bar = trange(n_iters, desc="[loss: NaN]")
        for _ in progress_bar:
            losses = []
            for i in np.random.permutation(len(draft_reduced_attentions)):
                x = draft_reduced_attentions[i].to(device, dtype)
                y = full_reduced_attentions[i].to(device, dtype)
                optimizer.zero_grad()
                loss = F.mse_loss(self(x), y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            scheduler.step()
            progress_bar.set_description(f"[loss: {sum(losses) / len(losses):.4e}]")

        self.w = self.w.cpu().detach()
        return self
