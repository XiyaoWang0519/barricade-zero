import random
import unittest

from barricade.agents import RandomAgent, ShortestPathAgent
from barricade.evaluation.arena import play_paired_match
from barricade.evaluation.openings import generate_opening_suite
from barricade.evaluation.players import AgentPlayerFactory


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


if __name__ == "__main__":
    unittest.main()
