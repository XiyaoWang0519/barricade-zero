import random
import subprocess
import sys
import unittest
from pathlib import Path

from barricade.backend import PythonRulesBackend, load_rules_backend
from barricade.state import GameState


class RulesBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        library = Path(__file__).parents[1] / "native" / "libbarricade_rules.so"
        if not library.exists():
            subprocess.run(
                [sys.executable, "scripts/build_native.py"],
                cwd=Path(__file__).parents[1],
                check=True,
                capture_output=True,
            )

    def test_python_fallback_matches_state_api(self):
        state = GameState.initial(size=9, walls_per_player=10)
        backend = load_rules_backend(prefer_native=False)
        self.assertIsInstance(backend, PythonRulesBackend)
        self.assertEqual(backend.legal_actions(state), state.legal_actions())
        self.assertEqual(
            backend.has_path_with_extra_wall(state, 0, "H", 3, 3),
            state._has_path_with_extra_wall(0, "H", 3, 3),
        )

    def test_native_backend_matches_python_on_random_reachable_states(self):
        from barricade.backend import NativeRulesBackend

        native = NativeRulesBackend()
        python = PythonRulesBackend()
        rng = random.Random(103)
        for size, walls in ((5, 4), (9, 10)):
            state = GameState.initial(size=size, walls_per_player=walls)
            for _ in range(25):
                expected = python.legal_actions(state)
                self.assertEqual(native.legal_actions(state), expected)
                for orientation, row, col in (("H", 0, 0), ("V", size - 2, size - 2)):
                    for player in (0, 1):
                        self.assertEqual(
                            native.has_path_with_extra_wall(
                                state, player, orientation, row, col
                            ),
                            python.has_path_with_extra_wall(
                                state, player, orientation, row, col
                            ),
                        )
                if not expected:
                    break
                state = state.apply_action(rng.choice(expected))
                if state.is_terminal():
                    break

    def test_prefer_native_returns_compiled_backend(self):
        backend = load_rules_backend(prefer_native=True)
        self.assertEqual(backend.name, "native")

    def test_native_backend_rejects_boards_larger_than_bitset_capacity(self):
        from barricade.backend import NativeRulesBackend

        with self.assertRaises(ValueError):
            NativeRulesBackend().legal_actions(
                GameState.initial(size=11, walls_per_player=10)
            )


if __name__ == "__main__":
    unittest.main()