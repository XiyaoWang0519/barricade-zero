import math
import random
import unittest

from barricade.actions import UP
from barricade.mcts import MCTS, UniformEvaluator
from barricade.state import GameState


class MCTSTests(unittest.TestCase):
    def test_search_only_assigns_probability_to_legal_actions(self):
        state = GameState.initial(size=5, walls_per_player=0)
        result = MCTS(UniformEvaluator(), simulations=32).search(state)
        self.assertEqual(sum(result.visits), 32)
        self.assertAlmostEqual(sum(result.policy), 1.0)
        legal = set(state.legal_actions())
        self.assertTrue(all(result.policy[action] == 0 for action in range(state.action_size) if action not in legal))

    def test_search_finds_immediate_win(self):
        state = GameState(size=5, pawns=((1, 0), (4, 4)), walls_remaining=(0, 0))
        result = MCTS(UniformEvaluator(), simulations=50).search(state)
        self.assertEqual(result.best_action, UP)

    def test_root_noise_is_reproducible_and_normalized(self):
        state = GameState.initial(size=5, walls_per_player=0)
        first = MCTS(UniformEvaluator(), simulations=8, dirichlet_alpha=0.3, noise_fraction=0.25, rng=random.Random(4)).search(state, add_noise=True)
        second = MCTS(UniformEvaluator(), simulations=8, dirichlet_alpha=0.3, noise_fraction=0.25, rng=random.Random(4)).search(state, add_noise=True)
        self.assertEqual(first.visits, second.visits)
        self.assertTrue(math.isclose(sum(first.root_priors), 1.0))

    def test_temperature_zero_returns_one_hot_policy(self):
        state = GameState.initial(size=5, walls_per_player=0)
        result = MCTS(UniformEvaluator(), simulations=16).search(state, temperature=0)
        self.assertEqual(sum(value == 1.0 for value in result.policy), 1)


if __name__ == "__main__":
    unittest.main()