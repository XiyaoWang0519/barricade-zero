import tempfile
import unittest
from pathlib import Path

from barricade.evaluation.openings import (
    generate_opening_suite,
    load_opening_suite,
    save_opening_suite,
)


class EvaluationOpeningTests(unittest.TestCase):
    def test_generation_is_reproducible_unique_and_non_terminal(self):
        first = generate_opening_suite(
            count=12,
            board_size=5,
            walls_per_player=2,
            min_plies=2,
            max_plies=5,
            seed=2026,
        )
        second = generate_opening_suite(
            count=12,
            board_size=5,
            walls_per_player=2,
            min_plies=2,
            max_plies=5,
            seed=2026,
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        keys = [opening.state.canonical_key() for opening in first.openings]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(not opening.state.is_terminal() for opening in first.openings))
        self.assertTrue(all(opening.state.legal_actions() for opening in first.openings))

    def test_suite_round_trips_with_stable_fingerprint(self):
        suite = generate_opening_suite(
            count=4,
            board_size=5,
            walls_per_player=0,
            min_plies=1,
            max_plies=3,
            seed=9,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "openings.json"
            save_opening_suite(path, suite)
            restored = load_opening_suite(path)
        self.assertEqual(restored.to_dict(), suite.to_dict())
        self.assertEqual(restored.sha256, suite.sha256)


if __name__ == "__main__":
    unittest.main()
