import random
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from barricade.backend import PythonRulesBackend, load_rules_backend
from barricade.encoding import encode_state
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

    def test_native_encoding_matches_python_on_random_reachable_states(self):
        from barricade.backend import NativeRulesBackend

        native = NativeRulesBackend()
        rng = random.Random(211)
        for size, walls in ((5, 4), (9, 10)):
            state = GameState.initial(size=size, walls_per_player=walls)
            for _ in range(20):
                canonical = state.canonical()
                self.assertEqual(
                    native.encode_state(canonical), encode_state(canonical)
                )
                actions = state.legal_actions()
                if not actions or state.is_terminal():
                    break
                state = state.apply_action(rng.choice(actions))

    def test_native_batch_encoding_is_contiguous_float32_and_matches_reference(self):
        from barricade.backend import NativeRulesBackend

        backend = NativeRulesBackend()
        self.assertTrue(hasattr(backend.library, "bz_encode_state_batch_f32"))
        states = [GameState.initial(size=9, walls_per_player=10)]
        states.append(states[0].apply_action(states[0].legal_actions()[0]).canonical())
        encoded = backend.encode_batch(states)
        self.assertEqual(encoded.shape, (2, 8, 9, 9))
        self.assertEqual(encoded.dtype, np.float32)
        self.assertTrue(encoded.flags.c_contiguous)
        expected = np.asarray([encode_state(state) for state in states], dtype=np.float32)
        np.testing.assert_array_equal(encoded, expected)

    def test_python_batch_encoding_has_same_contract(self):
        states = [GameState.initial(size=5, walls_per_player=4)] * 2
        encoded = PythonRulesBackend().encode_batch(states)
        self.assertEqual(encoded.shape, (2, 8, 5, 5))
        self.assertEqual(encoded.dtype, np.float32)
        self.assertTrue(encoded.flags.c_contiguous)

    def test_native_batch_legal_actions_csr_matches_individual_calls(self):
        from barricade.backend import NativeRulesBackend

        backend = NativeRulesBackend()
        state = GameState.initial(size=9, walls_per_player=10)
        states = [state]
        for index in range(7):
            legal = state.legal_actions()
            state = state.apply_action(legal[index % len(legal)])
            states.append(state)
        offsets, actions = backend.legal_actions_batch(states)
        self.assertEqual(offsets.dtype, np.int32)
        self.assertEqual(actions.dtype, np.int32)
        self.assertTrue(offsets.flags.c_contiguous)
        self.assertTrue(actions.flags.c_contiguous)
        self.assertEqual(offsets[0], 0)
        self.assertEqual(offsets[-1], len(actions))
        for index, state in enumerate(states):
            self.assertEqual(
                actions[offsets[index]:offsets[index + 1]].tolist(),
                backend.legal_actions(state),
            )

    def test_python_batch_legal_actions_has_same_contract(self):
        states = [GameState.initial(size=5, walls_per_player=2)] * 3
        offsets, actions = PythonRulesBackend().legal_actions_batch(states)
        self.assertEqual(offsets.shape, (4,))
        self.assertEqual(offsets.dtype, np.int32)
        self.assertEqual(actions.dtype, np.int32)

    def test_native_batch_legal_actions_matches_python_for_large_reachable_batch(self):
        from barricade.backend import NativeRulesBackend

        native = NativeRulesBackend()
        python = PythonRulesBackend()
        rng = random.Random(337)
        states = []
        for _ in range(128):
            state = GameState.initial(size=9, walls_per_player=10)
            for _ in range(rng.randrange(60)):
                legal = python.legal_actions(state)
                if not legal or state.is_terminal():
                    break
                state = state.apply_known_legal_action(rng.choice(legal))
            if not state.is_terminal():
                states.append(state)
        offsets, actions = native.legal_actions_batch(states)
        for index, state in enumerate(states):
            self.assertEqual(
                actions[offsets[index]:offsets[index + 1]].tolist(),
                python.legal_actions(state),
                f"mismatch at batch index {index}: {state}",
            )


if __name__ == "__main__":
    unittest.main()