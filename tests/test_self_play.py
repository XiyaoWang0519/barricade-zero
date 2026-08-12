import random
import unittest

from barricade.mcts import MCTS, UniformEvaluator
from barricade.self_play import play_self_play_game
from barricade.state import GameState


class SelfPlayTests(unittest.TestCase):
    def test_self_play_emits_normalized_policy_and_alternating_outcomes(self):
        mcts = MCTS(UniformEvaluator(), simulations=12, rng=random.Random(3))
        examples = play_self_play_game(
            GameState.initial(size=5, walls_per_player=0),
            mcts,
            rng=random.Random(4),
            exploration_plies=4,
        )
        self.assertTrue(examples)
        for example in examples:
            self.assertEqual(len(example.policy), 40)
            self.assertAlmostEqual(sum(example.policy), 1.0)
            self.assertIn(example.outcome, (-1.0, 1.0))
        for previous, current in zip(examples, examples[1:]):
            self.assertEqual(current.outcome, -previous.outcome)

    def test_recorded_states_use_canonical_perspective(self):
        mcts = MCTS(UniformEvaluator(), simulations=4, rng=random.Random(8))
        examples = play_self_play_game(
            GameState.initial(size=5, walls_per_player=0), mcts, rng=random.Random(9)
        )
        self.assertTrue(all(example.state.turn == 0 for example in examples))


if __name__ == "__main__":
    unittest.main()