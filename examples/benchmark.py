"""Benchmark HNSW against the exact flat index across dataset sizes.

Demonstrates the core trade-off of approximate search: build cost goes up, but
query latency stays nearly flat as the dataset grows. Run:

    python examples/benchmark.py
"""

import os
import random
import sys
import time

# Allow running the script directly from the examples/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lumen.flat import FlatIndex
from lumen.hnsw import HnswIndex

DIM = 32
SIZES = [1_000, 2_000]
QUERIES = 100


def make_vectors(n: int, dim: int, seed: int):
    rng = random.Random(seed)
    return [[rng.random() for _ in range(dim)] for _ in range(n)]


def bench(index, data, queries, k=10):
    t0 = time.perf_counter()
    for vec in data:
        index.add(f"v{len(index)}", vec)
    build = time.perf_counter() - t0

    t0 = time.perf_counter()
    for q in queries:
        index.search(q, k)
    query = time.perf_counter() - t0
    return build, query / QUERIES * 1000.0


print(f"{'index':<6} {'n':>7} {'build(s)':>9} {'ms/query':>10}")
for n in SIZES:
    data = make_vectors(n, DIM, seed=1)
    queries = make_vectors(QUERIES, DIM, seed=2)

    flat = FlatIndex(DIM, metric="euclidean")
    fb, fq = bench(flat, data, queries)

    hnsw = HnswIndex(DIM, metric="euclidean", M=16, ef_construction=200, ef=64, seed=1)
    hb, hq = bench(hnsw, data, queries)

    print(f"{'flat':<6} {n:>7} {fb:>9.2f} {fq:>10.3f}")
    print(f"{'hnsw':<6} {n:>7} {hb:>9.2f} {hq:>10.3f}")
    print()
