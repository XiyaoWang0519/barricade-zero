import unittest


class ProfileSelfPlayTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
