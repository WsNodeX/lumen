import random
import unittest

from lumen.flat import FlatIndex
from lumen.hnsw import HnswIndex


def random_vectors(n, dim, seed=1):
    rng = random.Random(seed)
    return [[rng.random() for _ in range(dim)] for _ in range(n)]


class TestHnswBasic(unittest.TestCase):
    def test_tiny_set_exact(self):
        idx = HnswIndex(dim=2, metric="euclidean", seed=1)
        idx.add("o", [0, 0])
        idx.add("a", [1, 0])
        idx.add("b", [0, 1])
        hits = idx.search([0, 0], k=1)
        self.assertEqual(hits[0].id, "o")

    def test_sorted_ascending(self):
        idx = HnswIndex(dim=3, metric="euclidean", seed=1)
        for i in range(20):
            idx.add(f"v{i}", [float(i % 2), float(i % 3), float(i % 5)])
        hits = idx.search([0, 0, 0], k=5)
        scores = [h.score for h in hits]
        self.assertEqual(scores, sorted(scores))
        self.assertLessEqual(len(hits), 5)

    def test_returns_valid_ids(self):
        vecs = random_vectors(100, 8, seed=2)
        idx = HnswIndex(dim=8, metric="euclidean", seed=3)
        for i, v in enumerate(vecs):
            idx.add(f"v{i}", v)
        hits = idx.search(vecs[0], k=10)
        valid = {f"v{i}" for i in range(100)}
        self.assertTrue(all(h.id in valid for h in hits))

    def test_delete(self):
        idx = HnswIndex(dim=2, metric="euclidean", seed=1)
        idx.add("k", [0, 0])
        idx.add("a", [1, 0])
        idx.add("b", [0, 1])
        self.assertTrue(idx.delete("k"))
        remaining = {h.id for h in idx.search([0.5, 0.5], k=5)}
        self.assertNotIn("k", remaining)
        self.assertEqual(len(idx), 2)


class TestHnswRecall(unittest.TestCase):
    def test_recall_vs_flat(self):
        vecs = random_vectors(500, 16, seed=11)
        flat = FlatIndex(dim=16, metric="euclidean")
        hnsw = HnswIndex(
            dim=16, metric="euclidean", M=16,
            ef_construction=200, ef=64, seed=11,
        )
        for i, v in enumerate(vecs):
            flat.add(f"v{i}", v)
            hnsw.add(f"v{i}", v)

        rng = random.Random(99)
        queries = [[rng.random() for _ in range(16)] for _ in range(50)]
        k = 10
        recalls = []
        for q in queries:
            truth = {h.id for h in flat.search(q, k)}
            got = {h.id for h in hnsw.search(q, k)}
            recalls.append(len(truth & got) / k)
        mean_recall = sum(recalls) / len(recalls)
        self.assertGreaterEqual(mean_recall, 0.85)

    def test_determinism_with_seed(self):
        vecs = random_vectors(200, 8, seed=5)
        a = HnswIndex(dim=8, metric="euclidean", seed=42)
        b = HnswIndex(dim=8, metric="euclidean", seed=42)
        for i, v in enumerate(vecs):
            a.add(f"v{i}", v)
            b.add(f"v{i}", v)
        q = [0.3, 0.1, 0.7, 0.2, 0.9, 0.4, 0.6, 0.8]
        ra = [h.id for h in a.search(q, k=10)]
        rb = [h.id for h in b.search(q, k=10)]
        self.assertEqual(ra, rb)


if __name__ == "__main__":
    unittest.main()
