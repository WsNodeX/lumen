"""Command-line interface for Lumen.

Exposes the most useful operations as a small, dependency-free CLI so the
database can be driven from shell scripts and pipelines:

    python -m lumen add     --db .lumen --name docs --id a --vec "1,0,0,0"
    python -m lumen search  --db .lumen --name docs --vec "0.9,0.1,0,0" --k 3
    python -m lumen bench   --dim 32 --n 2000 --queries 100 --index hnsw
    python -m lumen collections --db .lumen
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Dict, List

from .engine import Lumen
from .types import SearchResult


def _parse_vec(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip() != ""]


def _parse_meta(pairs: List[str]) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"--meta expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        meta[k] = v
    return meta


def cmd_add(args: argparse.Namespace) -> int:
    db = Lumen(args.db)
    vec = _parse_vec(args.vec)
    if args.name in db.collections():
        col = db.get_collection(args.name)
    else:
        col = db.create_collection(args.name, dim=len(vec), metric=args.metric)
    col.add(args.id, vec, metadata=_parse_meta(args.meta))
    path = db.persist(col)
    print(f"added {args.id} -> {path}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    db = Lumen(args.db)
    col = db.get_collection(args.name)
    vec = _parse_vec(args.vec)
    for hit in col.search(vec, k=args.k):
        print(f"{hit.id}\t{hit.score:.6f}\t{hit.metadata}")
    return 0


def cmd_collections(args: argparse.Namespace) -> int:
    db = Lumen(args.db)
    for name in db.collections():
        print(name)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    rng = random.Random(7)
    data = [[rng.random() for _ in range(args.dim)] for _ in range(args.n)]
    queries = [[rng.random() for _ in range(args.dim)] for _ in range(args.queries)]

    if args.index == "flat":
        from .flat import FlatIndex

        idx = FlatIndex(args.dim, metric=args.metric)
    else:
        from .hnsw import HnswIndex

        idx = HnswIndex(args.dim, metric=args.metric, seed=1)

    t0 = time.perf_counter()
    for i, vec in enumerate(data):
        idx.add(f"v{i}", vec)
    build_s = time.perf_counter() - t0

    k = min(10, args.n)
    t0 = time.perf_counter()
    total = 0
    for q in queries:
        total += len(idx.search(q, k))
    query_s = time.perf_counter() - t0

    print(f"index        : {args.index}")
    print(f"dim          : {args.dim}")
    print(f"vectors      : {args.n}")
    print(f"queries      : {args.queries} (k={k})")
    print(f"build time   : {build_s:.3f}s")
    print(f"query time   : {query_s:.3f}s "
          f"({1000 * query_s / max(args.queries, 1):.3f} ms/query)")
    print(f"results      : {total}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lumen", description="Lumen vector database CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="add a vector to a collection")
    p_add.add_argument("--db", default=".lumen")
    p_add.add_argument("--name", required=True)
    p_add.add_argument("--id", required=True)
    p_add.add_argument("--vec", required=True)
    p_add.add_argument("--metric", default="cosine")
    p_add.add_argument("--meta", action="append", default=[], help="key=value pairs")
    p_add.set_defaults(func=cmd_add)

    p_search = sub.add_parser("search", help="nearest-neighbour search")
    p_search.add_argument("--db", default=".lumen")
    p_search.add_argument("--name", required=True)
    p_search.add_argument("--vec", required=True)
    p_search.add_argument("--k", type=int, default=5)
    p_search.set_defaults(func=cmd_search)

    p_cols = sub.add_parser("collections", help="list collections")
    p_cols.add_argument("--db", default=".lumen")
    p_cols.set_defaults(func=cmd_collections)

    p_bench = sub.add_parser("bench", help="run a local benchmark")
    p_bench.add_argument("--dim", type=int, default=32)
    p_bench.add_argument("--n", type=int, default=2000)
    p_bench.add_argument("--queries", type=int, default=100)
    p_bench.add_argument("--index", choices=["flat", "hnsw"], default="hnsw")
    p_bench.add_argument("--metric", default="euclidean")
    p_bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
