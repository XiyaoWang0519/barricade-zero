import random
import unittest

from barricade.arena import Arena, ArenaResult
from barricade.mcts import UniformEvaluator


class ArenaTests(unittest.TestCase):
    def test_arena_balances_starting_sides(self):
        arena = Arena(simulations=4, board_size=5, walls_per_player=0, rng=random.Random(2))
        result = arena.play_match(UniformEvaluator(), UniformEvaluator(), games=4)
        self.assertIsInstance(result, ArenaResult)
        self.assertEqual(result.games, 4)
        self.assertEqual(result.candidate_wins + result.champion_wins + result.draws, 4)
        self.assertEqual(result.candidate_as_first_games, 2)
        self.assertEqual(result.candidate_as_second_games, 2)

    def test_arena_requires_even_game_count(self):
        arena = Arena(simulations=2, board_size=5, walls_per_player=0)
        with self.assertRaises(ValueError):
            arena.play_match(UniformEvaluator(), UniformEvaluator(), games=3)

    def test_batched_neural_arena_evaluates_multiple_games_together(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        torch.manual_seed(11)
        candidate = NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1))
        champion = NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1))
        arena = Arena(
            simulations=4,
            board_size=5,
            walls_per_player=0,
            rng=random.Random(3),
            max_plies=30,
        )
        result = arena.play_match(candidate, champion, games=8)
        self.assertEqual(result.games, 8)
        self.assertEqual(result.candidate_as_first_games, 4)
        self.assertGreater(max(candidate.batch_sizes), 1)
        self.assertGreater(candidate.average_batch_size, 1.0)


if __name__ == "__main__":
    unittest.main()