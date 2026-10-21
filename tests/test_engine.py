import os
import tempfile
import unittest

from lumen.engine import Lumen


class TestEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = Lumen(self.tmp)

    def test_create_and_list(self):
        self.db.create_collection("docs", dim=4)
        self.db.create_collection("images", dim=8)
        self.assertEqual(set(self.db.collections()), {"docs", "images"})

    def test_duplicate_name_raises(self):
        self.db.create_collection("docs", dim=2)
        with self.assertRaises(ValueError):
            self.db.create_collection("docs", dim=2)

    def test_persist_and_reload(self):
        col = self.db.create_collection("docs", dim=3, index_type="flat")
        col.add("a", [1, 0, 0], metadata={"t": "alpha"})
        col.add("b", [0, 1, 0])
        path = self.db.persist(col)
        self.assertTrue(os.path.exists(path))

        reloaded = self.db.get_collection("docs")
        self.assertEqual(len(reloaded), 2)
        self.assertEqual(reloaded.search([0.9, 0.1, 0], k=1)[0].id, "a")

    def test_get_missing_raises(self):
        with self.assertRaises(KeyError):
            self.db.get_collection("ghost")

    def test_drop(self):
        self.db.create_collection("docs", dim=2)
        self.assertTrue(self.db.drop_collection("docs"))
        self.assertFalse(self.db.drop_collection("docs"))
        self.assertEqual(self.db.collections(), [])

    def test_persist_all_and_stats(self):
        cols = []
        for name in ("a", "b"):
            c = self.db.create_collection(name, dim=2, index_type="flat")
            c.add("x", [1, 1])
            cols.append(c)
        self.db.persist_all(cols)
        stats = self.db.stats()
        self.assertEqual(set(stats), {"a", "b"})
        self.assertEqual(stats["a"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
