import unittest


class BenchmarkGenerationTests(unittest.TestCase):
    def test_summarize_runs_reports_phase_medians_and_dispersion(self):
        from scripts.benchmark_generation import PHASE_METRICS, summarize_runs

        runs = [
            {metric: float(index) for metric in PHASE_METRICS}
            for index in range(1, 6)
        ]

        summary = summarize_runs(runs)

        self.assertEqual(summary["repetitions"], 5)
        self.assertEqual(summary["generation_seconds"]["median"], 3.0)
        self.assertEqual(
            summary["self_play_seconds"]["median_absolute_deviation"], 1.0
        )
        self.assertEqual(summary["training_seconds"]["interquartile_range"], 2.0)
        self.assertGreater(
            summary["positions_per_generation_second"]["population_stddev"], 0.0
        )

    def test_empty_summary_is_rejected(self):
        from scripts.benchmark_generation import summarize_runs

        with self.assertRaisesRegex(ValueError, "at least one"):
            summarize_runs([])


if __name__ == "__main__":
    unittest.main()
