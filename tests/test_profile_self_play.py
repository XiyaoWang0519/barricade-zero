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


if __name__ == "__main__":
    unittest.main()
