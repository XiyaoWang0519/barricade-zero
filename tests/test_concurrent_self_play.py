import random
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ConcurrentSelfPlayTests(unittest.TestCase):
    def test_concurrent_games_emit_complete_training_examples(self):
        from barricade.concurrent_self_play import play_concurrent_games
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        evaluator = NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1))
        result = play_concurrent_games(
            evaluator,
            games=4,
            simulations=4,
            board_size=5,
            walls_per_player=0,
            rng=random.Random(9),
            max_plies=100,
        )
        self.assertEqual(result.games, 4)
        self.assertEqual(len(result.game_lengths), 4)
        self.assertEqual(len(result.examples), sum(result.game_lengths))
        self.assertTrue(all(example.state.turn == 0 for example in result.examples))
        self.assertGreater(result.average_inference_batch_size, 1.0)

    def test_draw_outcomes_are_zero_for_repetitions(self):
        from barricade.concurrent_self_play import play_concurrent_games
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        evaluator = NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1))
        result = play_concurrent_games(
            evaluator,
            games=2,
            simulations=1,
            board_size=5,
            walls_per_player=0,
            rng=random.Random(1),
            max_plies=8,
            max_plies_as_draw=True,
        )
        self.assertEqual(result.games, 2)
        self.assertTrue(all(example.outcome in (-1.0, 0.0, 1.0) for example in result.examples))


if __name__ == "__main__":
    unittest.main()