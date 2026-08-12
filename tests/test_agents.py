import random
import unittest

from barricade.agents import RandomAgent, ShortestPathAgent, play_game
from barricade.state import GameState


class BaselineAgentTests(unittest.TestCase):
    def test_random_agent_returns_a_legal_action(self):
        state = GameState.initial(size=5, walls_per_player=2)
        action = RandomAgent(random.Random(1)).choose_action(state)
        self.assertIn(action, state.legal_actions())

    def test_shortest_path_agent_reduces_distance_without_obstacles(self):
        state = GameState.initial(size=5, walls_per_player=0)
        action = ShortestPathAgent().choose_action(state)
        moved = state.apply_action(action)
        self.assertEqual(moved.shortest_path_distance(0), 3)

    def test_baseline_game_completes(self):
        winner, plies = play_game(
            ShortestPathAgent(), RandomAgent(random.Random(2)),
            GameState.initial(size=5, walls_per_player=2), max_plies=500,
        )
        self.assertIn(winner, (0, 1))
        self.assertLess(plies, 500)