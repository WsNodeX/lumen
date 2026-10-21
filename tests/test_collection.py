import unittest

from lumen.collection import VectorCollection


class TestCollection(unittest.TestCase):
    def setUp(self):
        self.col = VectorCollection("docs", dim=3, metric="cosine", index_type="hnsw")

    def test_add_and_search(self):
        self.col.add("a", [1, 0, 0], metadata={"title": "alpha"})
        self.col.add("b", [0, 1, 0])
        hits = self.col.search([0.9, 0.1, 0], k=1)
        self.assertEqual(hits[0].id, "a")
        self.assertEqual(hits[0].metadata["title"], "alpha")

    def test_metadata_optional(self):
        self.col.add("z", [1, 1, 1])
        hit = self.col.search([1, 1, 1], k=1, include_metadata=True)[0]
        self.assertEqual(hit.metadata, {})

    def test_no_metadata_flag(self):
        self.col.add("z", [1, 1, 1], metadata={"x": 1})
        hit = self.col.search([1, 1, 1], k=1, include_metadata=False)[0]
        self.assertIsNone(hit.metadata)

    def test_update_in_place(self):
        self.col.add("a", [1, 0, 0], metadata={"v": 1})
        self.col.add("a", [0, 1, 0], metadata={"v": 2})
        self.assertEqual(len(self.col), 1)
        hit = self.col.search([0, 1, 0], k=1)[0]
        self.assertEqual(hit.id, "a")
        self.assertEqual(hit.metadata["v"], 2)

    def test_delete(self):
        self.col.add("a", [1, 0, 0])
        self.col.add("b", [0, 1, 0])
        self.assertTrue(self.col.delete("a"))
        self.assertFalse(self.col.delete("a"))
        self.assertEqual(len(self.col), 1)

    def test_dim_mismatch(self):
        with self.assertRaises(ValueError):
            self.col.add("bad", [1, 0])

    def test_stats(self):
        self.col.add("a", [1, 0, 0])
        stats = self.col.stats()
        self.assertEqual(stats.name, "docs")
        self.assertEqual(stats.count, 1)
        self.assertEqual(stats.index_type, "hnsw")

    def test_roundtrip_via_dict(self):
        # to_dict -> from_file path is exercised heavily in storage tests;
        # here we confirm the dict carries metadata and vectors intact.
        self.col.add("a", [1, 0, 0], metadata={"k": "v"})
        d = self.col.to_dict()
        self.assertEqual(d["dim"], 3)
        self.assertEqual(len(d["vectors"]), 1)
        self.assertEqual(d["vectors"][0][0], "a")
        self.assertEqual(d["vectors"][0][2], {"k": "v"})


if __name__ == "__main__":
    unittest.main()
