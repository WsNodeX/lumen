import unittest

from lumen.flat import FlatIndex


class TestFlatIndex(unittest.TestCase):
    def setUp(self):
        self.idx = FlatIndex(dim=2, metric="euclidean")
        self.idx.add("a", [0, 0])
        self.idx.add("b", [3, 4])  # distance 5 from origin
        self.idx.add("c", [1, 0])  # distance 1 from origin

    def test_exact_nearest(self):
        hits = self.idx.search([0, 0], k=1)
        self.assertEqual(hits[0].id, "a")
        self.assertAlmostEqual(hits[0].score, 0.0, places=9)

    def test_ordering(self):
        hits = self.idx.search([0, 0], k=3)
        ids = [h.id for h in hits]
        self.assertEqual(ids, ["a", "c", "b"])
        scores = [h.score for h in hits]
        self.assertEqual(scores, sorted(scores))

    def test_k_larger_than_size(self):
        hits = self.idx.search([0, 0], k=99)
        self.assertEqual(len(hits), 3)

    def test_empty(self):
        self.assertEqual(FlatIndex(dim=3).search([0, 0, 0], k=5), [])

    def test_dim_mismatch_add(self):
        with self.assertRaises(ValueError):
            self.idx.add("x", [1, 2, 3])

    def test_dim_mismatch_search(self):
        with self.assertRaises(ValueError):
            self.idx.search([1, 2, 3], k=1)

    def test_bad_k(self):
        with self.assertRaises(ValueError):
            self.idx.search([0, 0], k=0)

    def test_delete(self):
        self.assertTrue(self.idx.delete("a"))
        self.assertFalse(self.idx.delete("a"))
        hits = self.idx.search([0, 0], k=3)
        self.assertEqual([h.id for h in hits], ["c", "b"])

    def test_len(self):
        self.assertEqual(len(self.idx), 3)


if __name__ == "__main__":
    unittest.main()
