import unittest

from barricade.evaluation.statistics import (
    confidence_decision,
    paired_bootstrap_interval,
)


class EvaluationStatisticsTests(unittest.TestCase):
    def test_paired_bootstrap_preserves_constant_scores(self):
        interval = paired_bootstrap_interval(
            [0.75] * 20,
            confidence=0.95,
            resamples=500,
            seed=7,
        )
        self.assertEqual(interval.mean, 0.75)
        self.assertEqual(interval.lower, 0.75)
        self.assertEqual(interval.upper, 0.75)
        self.assertEqual(interval.pairs, 20)

    def test_confidence_decision_requires_enough_opening_pairs(self):
        interval = paired_bootstrap_interval([1.0] * 8, resamples=200, seed=3)
        decision = confidence_decision(interval, minimum_pairs=10)
        self.assertEqual(decision.status, "insufficient_data")
        self.assertFalse(decision.eligible)

    def test_confidence_decision_can_promote_reject_or_remain_inconclusive(self):
        promoted = confidence_decision(
            paired_bootstrap_interval([0.75] * 20, resamples=200, seed=1),
            minimum_pairs=20,
        )
        rejected = confidence_decision(
            paired_bootstrap_interval([0.25] * 20, resamples=200, seed=1),
            minimum_pairs=20,
        )
        inconclusive = confidence_decision(
            paired_bootstrap_interval([0.0, 1.0] * 10, resamples=500, seed=1),
            minimum_pairs=20,
        )
        self.assertEqual(promoted.status, "promote")
        self.assertEqual(rejected.status, "reject")
        self.assertEqual(inconclusive.status, "inconclusive")

    def test_paired_bootstrap_is_deterministic_for_a_fixed_seed(self):
        scores = [index / 19 for index in range(20)]
        first = paired_bootstrap_interval(scores, resamples=2000, seed=11)
        second = paired_bootstrap_interval(scores, resamples=2000, seed=11)
        third = paired_bootstrap_interval(scores, resamples=2000, seed=12)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertNotEqual(first.lower, third.lower)


if __name__ == "__main__":
    unittest.main()
