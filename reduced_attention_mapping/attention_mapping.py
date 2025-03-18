from abc import ABC, abstractmethod
from typing import Optional, Self, TypeVar

from torch import Tensor

T = TypeVar("T", Tensor, list[Tensor])


class AttentionMapping(ABC):
    @abstractmethod
    def __init__(self, path: Optional[str] = None) -> None: ...

    @abstractmethod
    def fit(
        self,
        draft_reduced_attentions: list[Tensor],
        full_reduced_attentions: list[Tensor],
    ) -> Self: ...

    @abstractmethod
    def __call__(self, draft_reduced_attentions: T) -> T: ...

    @abstractmethod
    def save(self, path: str) -> None: ...
