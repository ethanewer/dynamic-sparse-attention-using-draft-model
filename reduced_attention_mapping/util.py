import warnings

import torch
from torch import Tensor
from torch.func import vmap  # type: ignore


def topk_overlap(
    pred_reduced_attentions: Tensor | list[Tensor],
    true_reduced_attentions: Tensor | list[Tensor],
    r: float = 0.125,
) -> float:
    if not isinstance(pred_reduced_attentions, list):
        pred_reduced_attentions = [pred_reduced_attentions]

    if not isinstance(true_reduced_attentions, list):
        true_reduced_attentions = [true_reduced_attentions]

    pred_topks = [
        a.cpu().topk(int(r * a.shape[-1])).indices.view(-1, int(r * a.shape[-1]))
        for a in pred_reduced_attentions
    ]
    true_topks = [
        a.topk(int(r * a.shape[-1])).indices.view(-1, int(r * a.shape[-1]))
        for a in true_reduced_attentions
    ]
    overlap = []
    for i in range(len(true_topks)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            overlap.append(
                vmap(torch.isin)(pred_topks[i], true_topks[i]).float().mean().item()
            )

    return sum(overlap) / len(overlap)