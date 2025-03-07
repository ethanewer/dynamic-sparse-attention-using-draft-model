from abc import ABC, abstractmethod
from typing import Optional, Self, TypeVar

import torch
from torch import Tensor

T = TypeVar("T", Tensor, list[Tensor])


class AttentionMapping(ABC):
    w: Optional[Tensor] = None

    def __init__(self, path: Optional[str] = None) -> None:
        if path is not None:
            parameters = torch.load(path)
            assert "w" in parameters
            self.w = parameters["w"]

    @abstractmethod
    def fit(
        self,
        draft_reduced_attentions: list[Tensor],
        full_reduced_attentions: list[Tensor],
    ) -> Self: ...

    def map_single(self, a: Tensor) -> Tensor:
        assert self.w is not None
        if a.dtype != self.w.dtype:
            self.w = self.w.to(dtype=a.dtype)

        if a.device != self.w.device:
            self.w = self.w.to(device=a.device)

        return torch.einsum("lbht,hlHL->LbHt", a, self.w)

    def __call__(self, draft_reduced_attentions: T) -> T:
        if isinstance(draft_reduced_attentions, list):
            return [self.map_single(a) for a in draft_reduced_attentions]
        else:
            return self.map_single(draft_reduced_attentions)

    def save(self, path: str) -> None:
        assert self.w is not None
        torch.save({"w": self.w}, path)
