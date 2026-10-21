"""Lumen — a zero-dependency vector database in pure Python.

Lumen stores fixed-dimension vectors, indexes them with either an exact
brute-force scan or an approximate HNSW graph, and persists every collection as
a single self-describing JSON document. It is designed to be embedded, tested and
shipped without a server.

Typical use
-----------
>>> from lumen import Lumen
>>> db = Lumen("./.lumen")
>>> docs = db.create_collection("docs", dim=4)
>>> docs.add("a", [1, 0, 0, 0], metadata={"title": "alpha"})
>>> docs.add("b", [0, 1, 0, 0])
>>> db.persist(docs)
>>> hits = db.get_collection("docs").search([0.9, 0.1, 0, 0], k=1)
>>> hits[0].id
'a'
"""

from __future__ import annotations

from .collection import VectorCollection
from .distance import METRICS, available_metrics, get_metric
from .engine import Lumen
from .flat import FlatIndex
from .hnsw import HnswIndex
from .index import VectorIndex
from .storage import FORMAT, VERSION
from .types import CollectionStats, SearchResult

__all__ = [
    "Lumen",
    "VectorCollection",
    "VectorIndex",
    "FlatIndex",
    "HnswIndex",
    "SearchResult",
    "CollectionStats",
    "get_metric",
    "available_metrics",
    "METRICS",
    "FORMAT",
    "VERSION",
]

__version__ = "0.1.0"
