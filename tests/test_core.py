import unittest
import numpy as np
from bplat.numerics import clean_mask, left_tail_crossing
from bplat.research_geometry import observed_quantile
from bplat.pipeline import local_prior, lookup


class CoreTests(unittest.TestCase):
    def test_observed_quantile_uses_a_real_pixel(self):
        values = np.arange(81, dtype=float).reshape(9, 9)
        valid = np.ones_like(values, dtype=bool)
        valid[0, 0] = False
        q40 = observed_quantile(values[valid], 0.4)
        self.assertEqual(q40, np.sort(values[valid])[int(np.floor(0.4 * (valid.sum() - 1)))])

    def test_left_tail_crossing_is_below_dark_mean(self):
        model = {"mu": np.array([-3.0, 0.0, 2.0]),
                 "var": np.array([0.4, 1.0, 1.0]),
                 "pi": np.array([0.2, 0.5, 0.3])}
        crossing = left_tail_crossing(model)
        self.assertIsNotNone(crossing)
        self.assertLess(crossing, model["mu"][0])

    def test_cleanup_removes_small_components(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[2:5, 2:5] = True
        mask[8, 8:10] = True
        result = clean_mask(mask, np.ones_like(mask), 5)
        self.assertFalse(result[8, 8])
        self.assertTrue(result[3, 3])

    def test_tail_override_increases_local_dark_prior(self):
        values = np.full((11, 11), -6.0)
        valid = np.ones_like(values, bool)
        model = {"mu": np.array([-3.0, 0.0, 2.0]),
                 "var": np.array([0.4, 1.0, 1.0]),
                 "pi": np.array([0.2, 0.5, 0.3])}
        without, _ = local_prior(values, valid, model, 1, "db", np.zeros_like(valid))
        with_tail, _ = local_prior(values, valid, model, 1, "db", valid)
        self.assertGreater(with_tail[5, 5], without[5, 5])
        self.assertGreater(with_tail[5, 5], 0.9)

    def test_lookup_threshold_is_monotone_in_prior(self):
        values = np.linspace(-10, 8, 100).reshape(10, 10)
        model = {"mu": np.array([-3.0, 0.0, 2.0]),
                 "var": np.array([0.4, 1.0, 1.0]),
                 "pi": np.array([0.2, 0.5, 0.3])}
        probabilities, thresholds = lookup(values, np.ones_like(values, bool), model, "db")
        self.assertEqual(len(probabilities), 2048)
        self.assertTrue(np.all(np.diff(thresholds) >= 0))


if __name__ == "__main__":
    unittest.main()
