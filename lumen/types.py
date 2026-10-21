"""Shared data types for Lumen.

A vector database lives or dies by the clarity of its data model. Lumen keeps
the model deliberately small: a *collection* holds a bag of fixed-dimension
vectors, each addressed by a stable string id and carrying an optional JSON
metadata payload. Search returns :class:`SearchResult` objects ordered by
ascending distance (smaller distance == more similar).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    """A single nearest-neighbour hit.

    ``score`` is the distance produced by the active metric. For every metric
    Lumen exposes, a *smaller* score means a *closer* neighbour, so callers can
    always sort ascending and read top-k without caring which metric is live.
    """

    id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"SearchResult(id={self.id!r}, score={self.score:.6f})"


@dataclass
class CollectionStats:
    """Cheap, read-only snapshot of a collection's shape."""

    name: str
    dim: int
    metric: str
    index_type: str
    count: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dim,
            "metric": self.metric,
            "index_type": self.index_type,
            "count": self.count,
        }
