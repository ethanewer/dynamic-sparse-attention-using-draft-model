from typing import Self

import torch
from torch import Tensor
from tqdm.notebook import trange  # type: ignore

from .attention_mapping import AttentionMapping


class GreedyAttentionMapping(AttentionMapping):
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
            a.topk(int(r * a.shape[-1])).indices
            for a in self.seperate_batch(draft_reduced_attentions)
        ]
        full_topks = [
            a.topk(int(r * a.shape[-1])).indices
            for a in self.seperate_batch(full_reduced_attentions)
        ]

        self.w = torch.zeros(
            num_draft_heads,
            num_draft_layers,
            num_full_heads,
            num_full_layers,
        )

        for i in trange(num_full_heads):
            for j in range(num_full_layers):
                scores = torch.zeros(num_draft_heads, num_draft_layers)
                for x, y in zip(draft_topks, full_topks):
                    scores += torch.isin(x, y[j, i]).float().sum(dim=2).T

                k_max, l_max = torch.unravel_index(scores.argmax(), scores.shape)
                self.w[k_max, l_max, i, j] = 1

        return self
