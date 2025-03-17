import warnings
from typing import Literal
import torch.nn.functional as F

import torch
from torch import Tensor
from torch.func import vmap  # type: ignore


def pool_reduced_attentions(
    reduced_attentions: Tensor,
    pooling: Literal["avg", "max"] = "avg",
    kernel_size: int = 5,
) -> Tensor:
    assert kernel_size % 2 == 1
    if kernel_size == 1:
        return reduced_attentions
    elif pooling == "mean":
        return F.avg_pool1d(
            reduced_attentions.view(-1, reduced_attentions.shape[-1]),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        ).view(reduced_attentions.shape)
    elif pooling == "max":
        return F.max_pool1d(
            reduced_attentions.view(-1, reduced_attentions.shape[-1]),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ).view(reduced_attentions.shape)
    else:
        raise NotImplementedError


def topk_overlap(
    pred_reduced_attentions: Tensor | list[Tensor],
    true_reduced_attentions: Tensor | list[Tensor],
    pooling: Literal["avg", "max"] = "avg",
    kernel_size: int = 1,
    r: float = 0.125,
) -> float:
    if not isinstance(pred_reduced_attentions, list):
        pred_reduced_attentions = [pred_reduced_attentions]

    if not isinstance(true_reduced_attentions, list):
        true_reduced_attentions = [true_reduced_attentions]

    pred_topks = [
        pool_reduced_attentions(a, pooling, kernel_size)
        .cpu()
        .topk(int(r * a.shape[-1]))
        .indices.view(-1, int(r * a.shape[-1]))
        for a in pred_reduced_attentions
    ]
    true_topks = [
        pool_reduced_attentions(a, pooling, kernel_size)
        .topk(int(r * a.shape[-1]))
        .indices.view(-1, int(r * a.shape[-1]))
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
