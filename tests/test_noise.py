import random
import unittest

import numpy as np

from barricade.noise import segmented_dirichlet_noise


class ExplorationNoiseTests(unittest.TestCase):
    def test_segmented_noise_is_reproducible_positive_and_normalized(self):
        lengths = [3, 7, 2]
        first = segmented_dirichlet_noise(random.Random(41), lengths, 0.3)
        second = segmented_dirichlet_noise(random.Random(41), lengths, 0.3)

        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first > 0))
        offset = 0
        for length in lengths:
            self.assertAlmostEqual(float(first[offset:offset + length].sum()), 1.0)
            offset += length

    def test_segmented_noise_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "segment lengths"):
            segmented_dirichlet_noise(random.Random(1), [2, 0], 0.3)
        with self.assertRaisesRegex(ValueError, "alpha"):
            segmented_dirichlet_noise(random.Random(1), [2], 0.0)


if __name__ == "__main__":
    unittest.main()
