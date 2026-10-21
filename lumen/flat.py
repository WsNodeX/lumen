"""Exact, brute-force nearest-neighbour index.

Scans every stored vector and keeps the k smallest distances in a bounded heap.
It is O(n) per query, which is hopeless at scale but *provably correct* — that
makes it the reference implementation against which the approximate index is
tested, and a sensible default for small collections and tests.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Tuple

from .distance import get_metric
from .index import VectorIndex
from .types import SearchResult


class FlatIndex(VectorIndex):
    def __init__(self, dim: int, metric: str = "cosine") -> None:
        if dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = dim
        self.metric_name = metric
        self._metric = get_metric(metric)
        self._vectors: Dict[str, List[float]] = {}

    def add(self, id: str, vector: List[float]) -> None:
        if len(vector) != self.dim:
            raise ValueError(
                f"vector has {len(vector)} dims, collection expects {self.dim}"
            )
        self._vectors[id] = list(vector)

    def search(self, vector: List[float], k: int) -> List[SearchResult]:
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if len(vector) != self.dim:
            raise ValueError(
                f"query has {len(vector)} dims, collection expects {self.dim}"
            )
        # Bounded max-heap of (-distance, id): the root is always the *farthest*
        # of the current top-k, so any closer candidate replaces it.
        heap: List = []
        for vid, vec in self._vectors.items():
            d = self._metric(vector, vec)
            if len(heap) < k:
                heapq.heappush(heap, (-d, vid))
            elif d < -heap[0][0]:
                heapq.heapreplace(heap, (-d, vid))
        ranked = sorted((-neg, vid) for neg, vid in heap)
        return [SearchResult(id=vid, score=d) for d, vid in ranked]

    def delete(self, id: str) -> bool:
        return self._vectors.pop(id, None) is not None

    def items(self) -> List[Tuple[str, List[float]]]:
        return list(self._vectors.items())

    def __len__(self) -> int:
        return len(self._vectors)
