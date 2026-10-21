"""Hierarchical Navigable Small World (HNSW) index.

HNSW is the family of algorithm that powers the "find nearest vectors" step in
most production embedding stores. The idea: instead of scanning every vector,
build a layered graph where the top layers are sparse "highways" for fast
long-range jumps and the bottom layer is dense for fine-grained local search.
A query walks down the layers, greedily hopping to the neighbour closest to the
query, and finishes with an exact-ish scan of a small candidate set.

This module is a from-scratch, dependency-free implementation. The public
surface is intentionally tiny (``add`` / ``search`` / ``delete``); the graph
bookkeeping lives in the private helpers.

Key parameters
--------------
``M``                  max connections kept per node on upper layers.
``M_max0``             2 * M connections on the base layer (empirically best).
``ef_construction``   candidate list size while *building* the graph.
``ef``                 candidate list size while *querying* (raise for recall).
``ml``                 layer-scale; derived as 1 / ln(M) by default.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .distance import get_metric
from .index import VectorIndex
from .types import SearchResult

Vector = List[float]


@dataclass
class _Node:
    id: str
    vector: List[float]
    level: int
    # neighbours[layer] is the set of node ids linked to this node at that layer.
    neighbours: List[Set[str]] = field(default_factory=list)


class HnswIndex(VectorIndex):
    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        M: int = 16,
        ef_construction: int = 200,
        ef: int = 64,
        ml: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be a positive integer")
        if M < 2:
            raise ValueError("M must be >= 2")
        self.dim = dim
        self.metric_name = metric
        self._metric = get_metric(metric)
        self.M = M
        self.M_max0 = 2 * M
        self.ef_construction = ef_construction
        self.ef = ef
        self.ml = ml if ml is not None else 1.0 / math.log(M)
        self._rng = random.Random(seed)
        self.nodes: Dict[str, _Node] = {}
        self.entry_point: Optional[str] = None
        self.max_level = -1

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add(self, id: str, vector: List[float]) -> None:
        if len(vector) != self.dim:
            raise ValueError(
                f"vector has {len(vector)} dims, index expects {self.dim}"
            )
        if id in self.nodes:
            # Simplest correct update: rebuild without the stale vector.
            self._rebuild_without(id)
        level = self._random_level()
        node = _Node(
            id=id,
            vector=list(vector),
            level=level,
            neighbours=[set() for _ in range(level + 1)],
        )
        self.nodes[id] = node

        if self.entry_point is None:
            self.entry_point = id
            self.max_level = level
            return

        # Greedy descent through the upper layers to find a good entry.
        curr = self.entry_point
        curr_dist = self._metric(vector, self.nodes[curr].vector)
        for lc in range(self.max_level, level, -1):
            curr, curr_dist = self._greedy(curr, curr_dist, vector, lc)

        # Connect the new node at every layer it participates in.
        top = min(level, self.max_level)
        for lc in range(top, -1, -1):
            candidates = self._search_layer(vector, [curr], self.ef_construction, lc)
            chosen = self._select_neighbours(candidates, self.M if lc > 0 else self.M_max0)
            node.neighbours[lc] = set(chosen)
            for nb in chosen:
                nb_node = self.nodes[nb]
                nb_node.neighbours[lc].add(id)
                cap = self.M if lc > 0 else self.M_max0
                if len(nb_node.neighbours[lc]) > cap:
                    kept = self._select_neighbours(
                        [
                            (self._metric(self.nodes[n].vector, nb_node.vector), n)
                            for n in nb_node.neighbours[lc]
                        ],
                        cap,
                    )
                    nb_node.neighbours[lc] = set(kept)

        if level > self.max_level:
            self.max_level = level
            self.entry_point = id

    def search(self, vector: List[float], k: int) -> List[SearchResult]:
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if len(vector) != self.dim:
            raise ValueError(
                f"query has {len(vector)} dims, index expects {self.dim}"
            )
        if self.entry_point is None:
            return []

        ef = max(self.ef, k)
        curr = self.entry_point
        curr_dist = self._metric(vector, self.nodes[curr].vector)
        for lc in range(self.max_level, 0, -1):
            curr, curr_dist = self._greedy(curr, curr_dist, vector, lc)

        candidates = self._search_layer(vector, [curr], ef, 0)
        candidates.sort(key=lambda x: x[0])
        return [SearchResult(id=nid, score=d) for d, nid in candidates[:k]]

    def delete(self, id: str) -> bool:
        if id not in self.nodes:
            return False
        self._rebuild_without(id)
        return True

    def items(self) -> List[Tuple[str, List[float]]]:
        return [(n.id, n.vector) for n in self.nodes.values()]

    def __len__(self) -> int:
        return len(self.nodes)

    # ------------------------------------------------------------------ #
    # Graph mechanics
    # ------------------------------------------------------------------ #
    def _random_level(self) -> int:
        # Exponential decay => most nodes sit on the base layer, a few climb
        # high. max(...) guards the impossible rng.random() == 0 case.
        u = max(self._rng.random(), 1e-12)
        return int(math.floor(-math.log(u) * self.ml))

    def _greedy(self, start: str, start_dist: float, q: Vector, layer: int):
        """Walk one layer towards ``q``, returning the closest node found."""
        curr, curr_dist = start, start_dist
        while True:
            improved = False
            for nb in self.nodes[curr].neighbours[layer]:
                d = self._metric(q, self.nodes[nb].vector)
                if d < curr_dist:
                    curr, curr_dist = nb, d
                    improved = True
            if not improved:
                return curr, curr_dist

    def _search_layer(
        self, q: Vector, entry_points: List[str], ef: int, layer: int
    ) -> List[Tuple[float, str]]:
        """Return up to ``ef`` (distance, id) pairs from one layer.

        A min-heap drives the candidate frontier (closest first) while a max-heap
        tracks the current best set; the loop stops once the nearest candidate
        can no longer beat the worst of the best.
        """
        visited: Set[str] = set(entry_points)
        candidates: List = []  # (distance, id) min-heap
        results: List = []  # (-distance, id) max-heap, top == farthest kept
        for ep in entry_points:
            d = self._metric(q, self.nodes[ep].vector)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))

        while candidates:
            d_c, c = heapq.heappop(candidates)
            # Farthest of the kept results is closer than the next candidate.
            if -results[0][0] < d_c:
                break
            for nb in self.nodes[c].neighbours[layer]:
                if nb not in visited:
                    visited.add(nb)
                    d_nb = self._metric(q, self.nodes[nb].vector)
                    if len(results) < ef or d_nb < -results[0][0]:
                        heapq.heappush(candidates, (d_nb, nb))
                        heapq.heappush(results, (-d_nb, nb))
                        if len(results) > ef:
                            heapq.heappop(results)
        return [(-neg, nid) for neg, nid in results]

    @staticmethod
    def _select_neighbours(candidates: List[Tuple[float, str]], m: int) -> List[str]:
        """Keep the ``m`` closest candidates (ascending distance)."""
        candidates.sort(key=lambda x: x[0])
        return [nid for _, nid in candidates[:m]]

    def _rebuild_without(self, drop_id: str) -> None:
        """Recreate the graph from all vectors except ``drop_id``.

        HNSW delete-with-repair is fiddly; for a lightweight library the honest,
        always-correct choice is to rebuild from the surviving vectors. The cost
        is O(n) per delete, which is perfectly fine at the scale Lumen targets.
        """
        survivors = [
            (n.id, n.vector) for n in self.nodes.values() if n.id != drop_id
        ]
        self.nodes = {}
        self.entry_point = None
        self.max_level = -1
        for vid, vec in survivors:
            # Reuse add(); it self-initialises when the graph is empty.
            HnswIndex.add(self, vid, vec)
