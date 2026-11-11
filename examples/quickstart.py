"""Quick start: build, query and persist a small collection in under 20 lines.

Run from the repository root:

    python examples/quickstart.py
"""

import os
import sys

# Allow running the script directly from the examples/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lumen import Lumen

# A fixed path made this example fail on the second run with "collection 'docs'
# already exists" — the first run left ./.lumen-demo behind. A fresh temp
# directory each time keeps the example repeatable and keeps the repo clean.
import tempfile

db = Lumen(tempfile.mkdtemp(prefix="lumen-quickstart-"))

# 1. Create a collection of 4-dimensional vectors.
docs = db.create_collection("docs", dim=4, metric="cosine")

# 2. Insert vectors with optional metadata.
docs.add("alpha", [1.0, 0.0, 0.0, 0.0], metadata={"title": "alpha"})
docs.add("beta", [0.0, 1.0, 0.0, 0.0], metadata={"title": "beta"})
docs.add("gamma", [0.0, 0.0, 1.0, 0.0], metadata={"title": "gamma"})

# 3. Persist to a single JSON document.
db.persist(docs)

# 4. Query the nearest neighbour of a near-alpha vector.
hits = db.get_collection("docs").search([0.9, 0.1, 0.0, 0.0], k=2)
for hit in hits:
    print(f"{hit.id:>6}  score={hit.score:.4f}  meta={hit.metadata}")
