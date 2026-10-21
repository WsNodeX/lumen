"""A named, queryable bag of vectors: the user-facing unit of Lumen.

A :class:`VectorCollection` owns one index (flat or HNSW) plus the metadata that
rides alongside each vector. It is the layer that turns raw index operations
into a friendly API: add with optional metadata, search returning rich results,
update-in-place by re-adding an id, and serialisation to / from the JSON shape
defined in :mod:`lumen.storage`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .flat import FlatIndex
from .hnsw import HnswIndex
from .storage import make_collection_dict, read_collection_file
from .types import CollectionStats, SearchResult

INDEX_FACTORIES = {
    "flat": FlatIndex,
    "hnsw": HnswIndex,
}

DEFAULT_HNSW_PARAMS: Dict[str, Any] = {
    "M": 16,
    "ef_construction": 200,
    "ef": 64,
}


class VectorCollection:
    def __init__(
        self,
        name: str,
        dim: int,
        metric: str = "cosine",
        index_type: str = "hnsw",
        index_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if index_type not in INDEX_FACTORIES:
            raise ValueError(
                f"unknown index_type {index_type!r}; choose from {sorted(INDEX_FACTORIES)}"
            )
        self.name = name
        self.dim = dim
        self.metric = metric
        self.index_type = index_type
        self._index_params = dict(index_params or {})
        self._meta: Dict[str, Dict[str, Any]] = {}
        self._index = self._make_index()

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def _make_index(self):
        factory = INDEX_FACTORIES[self.index_type]
        if self.index_type == "hnsw":
            params = dict(DEFAULT_HNSW_PARAMS)
            params.update(self._index_params)
            return factory(self.dim, metric=self.metric, **params)
        return factory(self.dim, metric=self.metric)

    @classmethod
    def from_file(cls, path: str) -> "VectorCollection":
        data = read_collection_file(path)
        col = cls(
            name=data["name"],
            dim=data["dim"],
            metric=data["metric"],
            index_type=data["index_type"],
            index_params=data.get("index_params", {}),
        )
        for vid, vector, meta in data["vectors"]:
            col._meta[vid] = meta or {}
            col._index.add(vid, vector)
        return col

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #
    def add(
        self, id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        if len(vector) != self.dim:
            raise ValueError(
                f"vector has {len(vector)} dims, collection expects {self.dim}"
            )
        self._meta[id] = dict(metadata or {})
        self._index.add(id, vector)

    def delete(self, id: str) -> bool:
        removed = self._index.delete(id)
        if removed:
            self._meta.pop(id, None)
        return removed

    def search(
        self, vector: List[float], k: int, include_metadata: bool = True
    ) -> List[SearchResult]:
        raw = self._index.search(vector, k)
        if not include_metadata:
            return raw
        return [
            SearchResult(
                id=r.id, score=r.score, metadata=self._meta.get(r.id)
            )
            for r in raw
        ]

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        records = [
            [vid, list(vec), self._meta.get(vid, {})]
            for vid, vec in self._index.items()
        ]
        return make_collection_dict(
            name=self.name,
            dim=self.dim,
            metric=self.metric,
            index_type=self.index_type,
            index_params=self._index_params,
            records=records,
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def stats(self) -> CollectionStats:
        return CollectionStats(
            name=self.name,
            dim=self.dim,
            metric=self.metric,
            index_type=self.index_type,
            count=len(self._index),
        )

    def __len__(self) -> int:
        return len(self._index)
