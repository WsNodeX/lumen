"""Distance and similarity primitives.

Every function here returns a *distance*: a non-negative-ish scalar where
*smaller means more similar*. That single invariant is what lets the rest of
Lumen treat cosine, Euclidean, Manhattan, Chebyshev and dot-product search
through one uniform code path.

All maths is plain Python stdlib so the package installs with zero
dependencies and runs on any interpreter from 3.9 upward.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List

Vector = List[float]


def _dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Vector) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_distance(a: Vector, b: Vector) -> float:
    """1 - cosine similarity. Range [0, 2]; 0 means identical direction."""
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        # A degenerate vector has no direction; treat it as maximally distant.
        return 1.0
    return 1.0 - _dot(a, b) / (na * nb)


def euclidean_distance(a: Vector, b: Vector) -> float:
    """Straight-line L2 distance."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def manhattan_distance(a: Vector, b: Vector) -> float:
    """L1 distance, the sum of absolute co-ordinate differences."""
    return sum(abs(x - y) for x, y in zip(a, b))


def chebyshev_distance(a: Vector, b: Vector) -> float:
    """L-infinity distance, the largest single co-ordinate gap."""
    return max(abs(x - y) for x, y in zip(a, b))


def dot_distance(a: Vector, b: Vector) -> float:
    """Negative dot product.

    Because search minimises distance, negating the dot product makes the
    closest neighbour the one with the *largest* dot product.
    """
    return -_dot(a, b)


METRICS: Dict[str, Callable[[Vector, Vector], float]] = {
    "cosine": cosine_distance,
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
    "chebyshev": chebyshev_distance,
    "dot": dot_distance,
}


def get_metric(name: str) -> Callable[[Vector, Vector], float]:
    """Resolve a metric by name, raising a clear error for typos."""
    if name not in METRICS:
        raise ValueError(
            f"Unknown metric {name!r}; available: {sorted(METRICS)}"
        )
    return METRICS[name]


def available_metrics() -> List[str]:
    return sorted(METRICS)
