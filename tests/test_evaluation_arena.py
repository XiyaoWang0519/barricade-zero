import random
import unittest

from barricade.agents import RandomAgent, ShortestPathAgent
from barricade.evaluation.arena import play_paired_match
from barricade.evaluation.openings import generate_opening_suite
from barricade.evaluation.players import AgentPlayerFactory, uniform_mcts_factory


class EvaluationArenaTests(unittest.TestCase):
    def test_match_plays_both_sides_of_every_opening(self):
        suite = generate_opening_suite(
            count=6,
            board_size=5,
            walls_per_player=0,
            min_plies=1,
            max_plies=3,
            seed=17,
        )
        candidate = AgentPlayerFactory(
            "shortest_path",
            lambda _seed: ShortestPathAgent(),
        )
        opponent = AgentPlayerFactory(
            "random",
            lambda seed: RandomAgent(random.Random(seed)),
        )
        result = play_paired_match(
            candidate,
            opponent,
            suite,
            max_plies=100,
            seed=81,
            bootstrap_resamples=500,
            minimum_pairs=6,
        )
        self.assertEqual(result.opening_pairs, 6)
        self.assertEqual(result.games, 12)
        self.assertEqual(result.wins + result.draws + result.losses, 12)
        self.assertEqual(len(result.pairs), 6)
        self.assertTrue(
            all(
                {game.candidate_player for game in pair.games} == {0, 1}
                for pair in result.pairs
            )
        )

    def test_match_is_reproducible_for_fixed_suite_and_seed(self):
        suite = generate_opening_suite(
            count=4,
            board_size=5,
            walls_per_player=0,
            min_plies=1,
            max_plies=2,
            seed=23,
        )

        def run():
            return play_paired_match(
                AgentPlayerFactory("shortest", lambda _seed: ShortestPathAgent()),
                AgentPlayerFactory(
                    "random", lambda seed: RandomAgent(random.Random(seed))
                ),
                suite,
                max_plies=100,
                seed=42,
                bootstrap_resamples=200,
                minimum_pairs=4,
            ).to_dict()

        self.assertEqual(run(), run())

    def test_search_matches_batch_inference_across_games(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        from barricade.evaluation.players import SearchPlayerFactory
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        suite = generate_opening_suite(
            count=4,
            board_size=5,
            walls_per_player=0,
            min_plies=2,
            max_plies=2,
            seed=29,
        )
        torch.manual_seed(8)
        candidate_evaluator = NeuralEvaluator(
            PolicyValueNetwork(5, channels=8, residual_blocks=1)
        )
        opponent_evaluator = NeuralEvaluator(
            PolicyValueNetwork(5, channels=8, residual_blocks=1)
        )
        result = play_paired_match(
            SearchPlayerFactory("candidate", candidate_evaluator, 4),
            SearchPlayerFactory("opponent", opponent_evaluator, 4),
            suite,
            max_plies=30,
            seed=13,
            bootstrap_resamples=200,
            minimum_pairs=4,
        )
        self.assertEqual(result.games, 8)
        self.assertGreater(max(candidate_evaluator.batch_sizes), 1)
        self.assertGreater(candidate_evaluator.average_batch_size, 1.0)

    def test_search_versus_agent_matches_keep_per_game_agent_state(self):
        suite = generate_opening_suite(
            count=4,
            board_size=5,
            walls_per_player=0,
            min_plies=1,
            max_plies=2,
            seed=31,
        )
        result = play_paired_match(
            uniform_mcts_factory(4),
            AgentPlayerFactory(
                "random", lambda seed: RandomAgent(random.Random(seed))
            ),
            suite,
            max_plies=40,
            seed=19,
            bootstrap_resamples=200,
            minimum_pairs=4,
        )
        self.assertEqual(result.games, 8)
        self.assertEqual(result.wins + result.draws + result.losses, 8)


if __name__ == "__main__":
    unittest.main()
