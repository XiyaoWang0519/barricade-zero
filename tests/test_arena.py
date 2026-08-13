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


if __name__ == "__main__":
    unittest.main()