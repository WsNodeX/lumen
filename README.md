<div align="center">

# Lumen

### A zero-dependency vector database in pure Python

Store embeddings, index them with an HNSW graph, query nearest neighbours, and
persist everything as a single JSON document — no server, no native build, no
third-party packages.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-0-lightgrey.svg)](pyproject.toml)
[![Platform](https://img.shields.io/badge/platform-cross--platform-cyan.svg)](https://github.com/WsNodeX/lumen)

**[Why Lumen](#why-lumen) ·
[Features](#features) ·
[Installation](#installation) ·
[Quick Start](#quick-start) ·
[Architecture](#architecture) ·
[API](#api) ·
[Roadmap](#roadmap) ·
[Contributing](#contributing)**

</div>

---

## ✨ Why Lumen

Embedding search is everywhere — retrieval-augmented generation, duplicate
detection, recommendation, semantic memory — yet the libraries that power it are
usually heavyweight services you have to stand up and operate. Lumen takes the
opposite tack: it is a **library you import**, not a system you deploy.

- **No dependencies.** The whole thing is standard-library Python. Drop it into
  any environment, from a laptop to a locked-down CI runner.
- **No server.** A collection is one JSON file. Commit it, copy it, email it.
- **Real algorithms.** The approximate index is a from-scratch
  Hierarchical Navigable Small World (HNSW) graph — the same family of
  algorithm that underpins industrial vector stores — not a toy.
- **Small and readable.** Every layer is a few hundred lines, so you can read
  the whole thing in an afternoon and actually understand how it works.

## 🚀 Features

- 🔎 **Exact and approximate search** — a brute-force `FlatIndex` for
  correctness/reference and an `HnswIndex` for speed at scale.
- 📐 **Five distance metrics** — `cosine`, `euclidean`, `manhattan`,
  `chebyshev`, and `dot` (maximise dot product).
- 🗂️ **Collections with metadata** — attach a JSON payload to every vector and
  get it back with each search hit.
- 💾 **Versioned JSON persistence** — each collection is a self-describing
  document, atomically written.
- 🧰 **Multi-collection engine** — a file-rooted `Lumen` database manages many
  named collections.
- ⌨️ **CLI** — add, search, list and benchmark from the terminal.
- ✅ **Fully tested** — the suite verifies distances, exact results, HNSW recall
  against brute force, persistence round-trips and the engine lifecycle.

## 📦 Installation

Lumen has no dependencies, so you can vendor it directly:

```bash
# clone and use as a package
git clone https://github.com/WsNodeX/lumen.git
cd lumen

# or install from source
pip install .
```

That is the entire setup. Python 3.9+ is the only requirement.

## ⚡ Quick Start

```python
from lumen import Lumen

db = Lumen("./.lumen-demo")

# Create a collection of 4-dimensional vectors.
docs = db.create_collection("docs", dim=4, metric="cosine")

# Insert vectors with optional metadata.
docs.add("alpha", [1.0, 0.0, 0.0, 0.0], metadata={"title": "alpha"})
docs.add("beta",  [0.0, 1.0, 0.0, 0.0], metadata={"title": "beta"})
docs.add("gamma", [0.0, 0.0, 1.0, 0.0], metadata={"title": "gamma"})

# Persist to a single JSON document.
db.persist(docs)

# Query the nearest neighbour of a near-alpha vector.
hits = db.get_collection("docs").search([0.9, 0.1, 0.0, 0.0], k=2)
for hit in hits:
    print(hit.id, round(hit.score, 4), hit.metadata)
```

**Expected output:**

```
alpha 0.0061 {'title': 'alpha'}
beta 0.8896 {'title': 'beta'}
```

### Command line

```bash
# add vectors
python -m lumen add --db .lumen --name docs --id a --vec "1,0,0,0" --meta title=alpha
python -m lumen add --db .lumen --name docs --id b --vec "0,1,0,0"

# search
python -m lumen search --db .lumen --name docs --vec "0.9,0.1,0,0" --k 2

# local benchmark
python -m lumen bench --dim 32 --n 2000 --queries 100 --index hnsw
```

## 🏗️ Architecture

Lumen is a small stack of single-responsibility layers. Each layer depends only
on the one below it, which keeps the code easy to test and to swap.

```
            ┌─────────────────────────────────────────────┐
            │                CLI  (lumen.cli)              │   terminal entry point
            └───────────────────────────┬─────────────────┘
                                        │
            ┌───────────────────────────┴─────────────────┐
            │            Engine  (lumen.engine)            │   multi-collection DB, file rooted
            └───────────────────────────┬─────────────────┘
                                        │
            ┌───────────────────────────┴─────────────────┐
            │        Collection  (lumen.collection)        │   named bag + metadata + index
            └──────────────┬───────────────────┬──────────┘
                           │                   │
            ┌──────────────┴──────┐  ┌─────────┴──────────┐
            │  FlatIndex (exact)  │  │  HnswIndex (ANN)    │   VectorIndex implementations
            └──────────────┬──────┘  └─────────┬──────────┘
                           │                   │
            ┌──────────────┴───────────────────┴──────────┐
            │          Distance  (lumen.distance)          │   cosine / euclidean / ...
            └───────────────────────────┬─────────────────┘
                                        │
            ┌───────────────────────────┴─────────────────┐
            │          Storage  (lumen.storage)            │   versioned JSON documents
            └─────────────────────────────────────────────┘
```

| Layer | Module | Responsibility |
|-------|--------|----------------|
| CLI | `lumen.cli` | Parse arguments; expose add / search / bench / collections. |
| Engine | `lumen.engine` | Manage a directory of named collections; persist and reload. |
| Collection | `lumen.collection` | Bind an index to a name, dimension, metric and metadata store. |
| Index | `lumen.flat`, `lumen.hnsw` | Implement the `VectorIndex` contract: add / search / delete. |
| Distance | `lumen.distance` | Pure-function metrics; every metric returns smaller = closer. |
| Storage | `lumen.storage` | Serialise / deserialise versioned JSON collection files. |

## 🔌 API

### `Lumen(root)` — the database

| Method | Signature | Description |
|--------|-----------|-------------|
| create | `create_collection(name, dim, metric="cosine", index_type="hnsw", index_params=None)` | Create and persist an empty collection. |
| get | `get_collection(name)` | Load a persisted collection by name. |
| list | `collections()` | List collection names on disk. |
| drop | `drop_collection(name)` | Delete a collection file. |
| persist | `persist(collection)` | Write a collection document; returns its path. |
| stats | `stats()` | Per-collection summary (dim, metric, count, …). |

### `VectorCollection` — the queryable unit

| Method | Signature | Description |
|--------|-----------|-------------|
| add | `add(id, vector, metadata=None)` | Insert or replace a vector and its metadata. |
| search | `search(vector, k, include_metadata=True)` | Return the `k` nearest `SearchResult`s. |
| delete | `delete(id)` | Remove a vector; returns `True` if one was removed. |
| stats | `stats()` | Return a `CollectionStats` snapshot. |

`SearchResult` exposes `.id`, `.score` (distance, smaller = closer) and
`.metadata`.

### Index selection

| `index_type` | Class | Use when |
|--------------|-------|----------|
| `"flat"` | `FlatIndex` | Small datasets or when 100% exact results are required. |
| `"hnsw"` | `HnswIndex` | Larger datasets where sub-linear query time matters. |

`HnswIndex` accepts `M` (connections per node), `ef_construction` (graph-build
candidate list), `ef` (query candidate list) and `seed` (reproducibility).

### Distance metrics

| Name | Meaning |
|------|---------|
| `cosine` | 1 − cosine similarity. |
| `euclidean` | Straight-line L2 distance. |
| `manhattan` | L1 sum of absolute differences. |
| `chebyshev` | L∞ largest single-axis gap. |
| `dot` | Negative dot product (search maximises the dot product). |

## 🧭 Roadmap

- [ ] Quantised (PQ/SQ) vectors to shrink memory at large scale
- [ ] Hybrid lexical + vector scoring
- [ ] Range search (`search_by_radius`)
- [ ] Async / concurrent index build
- [ ] Optional mmap-backed storage for collections larger than RAM
- [ ] `lumen serve` HTTP endpoint with OpenAPI docs

## 🤝 Contributing

Contributions are welcome. The project is intentionally small, so the fastest
path is:

1. Fork and create a feature branch.
2. Add or extend tests under `tests/` — the suite must stay green
   (`python -m unittest discover -s tests`).
3. Keep new code dependency-free and typed.
4. Open a pull request describing the change and the use case.

## 📄 License

Released under the [MIT License](LICENSE).
