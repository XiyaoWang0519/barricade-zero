import unittest

from barricade.backend import PythonRulesBackend, load_rules_backend
from barricade.state import GameState


class RulesBackendTests(unittest.TestCase):
    def test_python_fallback_matches_state_api(self):
        state = GameState.initial(size=9, walls_per_player=10)
        backend = load_rules_backend(prefer_native=False)
        self.assertIsInstance(backend, PythonRulesBackend)
        self.assertEqual(backend.legal_actions(state), state.legal_actions())
        self.assertEqual(
            backend.has_path_with_extra_wall(state, 0, "H", 3, 3),
            state._has_path_with_extra_wall(0, "H", 3, 3),
        )


if __name__ == "__main__":
    unittest.main()