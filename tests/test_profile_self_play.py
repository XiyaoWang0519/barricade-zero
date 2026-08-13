import unittest


class ProfileSelfPlayTests(unittest.TestCase):
    def test_summary_records_explicit_torch_thread_count(self):
        import torch

        from scripts.profile_self_play import run_workload

        previous = torch.get_num_threads()
        try:
            summary = run_workload(
                board_size=5,
                walls=0,
                games=1,
                simulations=1,
                seed=92,
                channels=8,
                residual_blocks=1,
                torch_threads=1,
            )
            self.assertEqual(summary["torch_threads"], 1)
            self.assertEqual(torch.get_num_threads(), 1)
        finally:
            torch.set_num_threads(previous)

    def test_summarize_runs_reports_median_and_dispersion(self):
        from scripts.profile_self_play import summarize_runs

        runs = [
            {
                "seconds": seconds,
                "model_forward_seconds": seconds / 2,
                "non_model_seconds": seconds / 2,
                "positions_per_second": 100.0 / seconds,
                "examples_per_second": 50.0 / seconds,
                "positions": 100,
                "examples": 50,
                "forward_calls": 10,
                "average_batch_size": 10.0,
            }
            for seconds in (1.0, 2.0, 3.0, 4.0, 5.0)
        ]

        summary = summarize_runs(runs)

        self.assertEqual(summary["repetitions"], 5)
        self.assertEqual(summary["seconds"]["median"], 3.0)
        self.assertGreater(summary["seconds"]["population_stddev"], 0.0)
        self.assertEqual(summary["seconds"]["median_absolute_deviation"], 1.0)
        self.assertEqual(summary["seconds"]["interquartile_range"], 2.0)
        self.assertEqual(summary["positions"]["median"], 100)

    def test_summary_records_requested_network_shape(self):
        from scripts.profile_self_play import run_workload

        summary = run_workload(
            board_size=5,
            walls=0,
            games=1,
            simulations=1,
            seed=91,
            channels=8,
            residual_blocks=1,
            collect_metrics=True,
        )
        self.assertEqual(summary["channels"], 8)
        self.assertEqual(summary["residual_blocks"], 1)
        self.assertEqual(summary["device"], "cpu")
        self.assertEqual(summary["precision"], "float32")
        self.assertIsNone(summary["gpu_model"])
        self.assertEqual(summary["peak_gpu_allocated_bytes"], 0)
        self.assertEqual(summary["peak_gpu_reserved_bytes"], 0)
        self.assertEqual(summary["gpu_utilization_samples"], 0)
        self.assertGreaterEqual(summary["model_forward_seconds"], 0.0)
        self.assertAlmostEqual(
            summary["non_model_seconds"],
            summary["seconds"] - summary["model_forward_seconds"],
        )
        self.assertAlmostEqual(
            summary["examples_per_second"],
            summary["examples"] / summary["seconds"],
        )
        self.assertEqual(summary["game_length_count"], 1)
        self.assertEqual(summary["game_length_median"], summary["examples"])
        self.assertGreater(summary["inference_batch_size_count"], 0)
        self.assertGreaterEqual(
            summary["inference_batch_size_max"],
            summary["inference_batch_size_median"],
        )
        self.assertIn(summary["rules_backend"], ("native", "python"))
        self.assertGreater(summary["python_native_boundary_calls"], 0)
        self.assertGreaterEqual(summary["legal_action_seconds"], 0.0)
        self.assertGreaterEqual(summary["encoding_seconds"], 0.0)
        self.assertGreater(summary["tree_nodes_created"], 0)
        self.assertGreater(summary["tree_expansions"], 0)
        self.assertGreaterEqual(summary["tree_traversal_seconds"], 0.0)
        self.assertGreaterEqual(summary["host_to_device_seconds"], 0.0)
        self.assertGreaterEqual(summary["device_synchronization_seconds"], 0.0)
        self.assertGreater(summary["peak_process_rss_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
