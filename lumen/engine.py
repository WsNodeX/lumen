"""The engine: a tiny multi-collection database rooted at a directory.

Think of :class:`Lumen` as a file-system database. Each collection is one JSON
document under a root directory; the engine creates, lists, loads, persists and
drops them. There is no server, no socket and no daemon — open the directory,
work with it, and persist when you choose. That makes Lumen trivial to embed,
snapshot and version-control.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .collection import VectorCollection
from .storage import (
    collection_path,
    list_collection_files,
    write_collection_file,
)


class Lumen:
    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Collection lifecycle
    # ------------------------------------------------------------------ #
    def create_collection(
        self,
        name: str,
        dim: int,
        metric: str = "cosine",
        index_type: str = "hnsw",
        index_params: Optional[Dict[str, Any]] = None,
    ) -> VectorCollection:
        if name in self.collections():
            raise ValueError(f"collection {name!r} already exists in {self.root}")
        col = VectorCollection(
            name=name,
            dim=dim,
            metric=metric,
            index_type=index_type,
            index_params=index_params,
        )
        # Materialise the empty collection on disk so the database reflects it
        # immediately; later mutations are flushed via ``persist``.
        self.persist(col)
        return col

    def get_collection(self, name: str) -> VectorCollection:
        path = collection_path(self.root, name)
        if not os.path.exists(path):
            raise KeyError(f"collection {name!r} not found in {self.root}")
        return VectorCollection.from_file(path)

    def collections(self) -> List[str]:
        files = list_collection_files(self.root)
        return [os.path.basename(f)[: -len(".lumen.json")] for f in files]

    def drop_collection(self, name: str) -> bool:
        path = collection_path(self.root, name)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def persist(self, collection: VectorCollection) -> str:
        """Write a collection document and return its absolute path."""
        path = collection_path(self.root, collection.name)
        write_collection_file(path, collection.to_dict())
        return path

    def persist_all(self, collections: List[VectorCollection]) -> List[str]:
        return [self.persist(c) for c in collections]

    def stats(self) -> Dict[str, Any]:
        out = {}
        for name in self.collections():
            try:
                out[name] = self.get_collection(name).stats().as_dict()
            except Exception as exc:  # pragma: no cover - defensive
                out[name] = {"error": str(exc)}
        return out
