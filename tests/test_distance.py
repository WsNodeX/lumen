import math
import unittest

from lumen.distance import (
    chebyshev_distance,
    cosine_distance,
    dot_distance,
    euclidean_distance,
    get_metric,
    manhattan_distance,
)


class TestDistance(unittest.TestCase):
    def test_cosine_identical(self):
        self.assertAlmostEqual(cosine_distance([1, 0], [1, 0]), 0.0, places=9)

    def test_cosine_orthogonal(self):
        self.assertAlmostEqual(cosine_distance([1, 0], [0, 1]), 1.0, places=9)

    def test_cosine_zero_vector(self):
        self.assertAlmostEqual(cosine_distance([0, 0], [1, 1]), 1.0, places=9)

    def test_euclidean(self):
        self.assertAlmostEqual(euclidean_distance([0, 0], [3, 4]), 5.0, places=9)

    def test_manhattan(self):
        self.assertAlmostEqual(manhattan_distance([0, 0], [3, 4]), 7.0, places=9)

    def test_chebyshev(self):
        self.assertAlmostEqual(chebyshev_distance([1, 5], [4, 1]), 4.0, places=9)

    def test_dot(self):
        # Maximising the dot product means minimising the negated value.
        self.assertAlmostEqual(dot_distance([1, 2], [3, 4]), -11.0, places=9)

    def test_get_metric_unknown(self):
        with self.assertRaises(ValueError):
            get_metric("nope")

    def test_get_metric_resolves(self):
        self.assertIs(get_metric("cosine"), cosine_distance)

    def test_symmetry(self):
        a, b = [0.1, -0.4, 0.9], [-0.3, 0.2, 0.5]
        for fn in (
            cosine_distance,
            euclidean_distance,
            manhattan_distance,
            chebyshev_distance,
            dot_distance,
        ):
            self.assertAlmostEqual(fn(a, b), fn(b, a), places=9)


if __name__ == "__main__":
    unittest.main()
