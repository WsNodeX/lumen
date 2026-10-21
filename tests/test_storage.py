import json
import os
import tempfile
import unittest

from lumen.collection import VectorCollection
from lumen.storage import (
    FORMAT,
    VERSION,
    collection_path,
    list_collection_files,
    read_collection_file,
    write_collection_file,
)


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_roundtrip_collection(self):
        col = VectorCollection("items", dim=3, metric="cosine", index_type="flat")
        col.add("x", [1, 0, 0], metadata={"label": "X"})
        col.add("y", [0, 1, 0], metadata={"label": "Y"})
        path = collection_path(self.tmp, col.name)
        write_collection_file(path, col.to_dict())

        self.assertTrue(os.path.exists(path))
        data = read_collection_file(path)
        self.assertEqual(data["format"], FORMAT)
        self.assertEqual(data["version"], VERSION)
        self.assertEqual(data["name"], "items")
        self.assertEqual(len(data["vectors"]), 2)

        reloaded = VectorCollection.from_file(path)
        self.assertEqual(len(reloaded), 2)
        self.assertEqual(reloaded.search([0.9, 0.1, 0], k=1)[0].id, "x")
        self.assertEqual(reloaded.search([0.9, 0.1, 0], k=1)[0].metadata["label"], "X")

    def test_list_collection_files(self):
        for n in ("a", "b"):
            col = VectorCollection(n, dim=2, index_type="flat")
            col.add("v", [1, 1])
            write_collection_file(collection_path(self.tmp, n), col.to_dict())
        files = list_collection_files(self.tmp)
        self.assertEqual(len(files), 2)
        self.assertTrue(all(f.endswith(".lumen.json") for f in files))

    def test_reject_non_lumen_file(self):
        bad = os.path.join(self.tmp, "bad.lumen.json")
        with open(bad, "w", encoding="utf-8") as fh:
            json.dump({"format": "something-else"}, fh)
        with self.assertRaises(ValueError):
            read_collection_file(bad)


if __name__ == "__main__":
    unittest.main()
