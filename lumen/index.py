"""Index interface.

Both the exact :class:`FlatIndex` and the approximate :class:`HnswIndex`
implement this contract, so the collection layer can swap implementations
without knowing the internals. The rule is simple: ``add`` grows the index,
``search`` returns the k nearest by distance, ``delete`` removes a vector, and
``__len__`` reports how many vectors are stored.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from .types import SearchResult


class VectorIndex(ABC):
    dim: int
    metric_name: str

    @abstractmethod
    def add(self, id: str, vector: List[float]) -> None:
        """Insert or replace ``vector`` under ``id``."""

    @abstractmethod
    def search(self, vector: List[float], k: int) -> List[SearchResult]:
        """Return the ``k`` nearest neighbours ordered by ascending distance."""

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Remove ``id`` if present; return True when something was removed."""

    @abstractmethod
    def items(self) -> List[Tuple[str, List[float]]]:
        """Return every (id, vector) pair currently stored."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of vectors currently indexed."""
