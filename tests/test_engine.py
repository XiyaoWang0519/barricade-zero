import random
import unittest
import json

from barricade.actions import (
    DOWN,
    DOWN_LEFT,
    DOWN_RIGHT,
    LEFT,
    RIGHT,
    UP,
    UP_LEFT,
    UP_RIGHT,
    decode_wall_action,
    encode_wall_action,
    rotate_action,
)
from barricade.state import GameState, IllegalAction


class ActionEncodingTests(unittest.TestCase):
    def test_official_board_has_136_actions_and_wall_round_trips(self):
        state = GameState.initial()
        self.assertEqual(state.action_size, 136)
        for orientation in ("H", "V"):
            for row in range(8):
                for col in range(8):
                    action = encode_wall_action(orientation, row, col, 9)
                    self.assertEqual(decode_wall_action(action, 9), (orientation, row, col))

    def test_rotating_an_action_twice_restores_it(self):
        for action in range(136):
            self.assertEqual(rotate_action(rotate_action(action, 9), 9), action)


class InitialStateTests(unittest.TestCase):
    def test_initial_state(self):
        state = GameState.initial()
        self.assertEqual(state.pawns, ((8, 4), (0, 4)))
        self.assertEqual(state.walls_remaining, (10, 10))
        self.assertEqual(state.turn, 0)
        self.assertEqual(set(state.legal_pawn_actions()), {UP, LEFT, RIGHT})
        self.assertFalse(state.is_terminal())
        self.assertTrue(state.has_path(0))
        self.assertTrue(state.has_path(1))

    def test_apply_move_is_immutable_and_switches_turn(self):
        state = GameState.initial()
        moved = state.apply_action(UP)
        self.assertEqual(state.pawns[0], (8, 4))
        self.assertEqual(moved.pawns[0], (7, 4))
        self.assertEqual(moved.turn, 1)


class PawnMovementTests(unittest.TestCase):
    def test_wall_blocks_ordinary_movement(self):
        state = GameState(
            size=5,
            pawns=((3, 2), (0, 2)),
            horizontal_walls=frozenset({(2, 2)}),
            walls_remaining=(2, 2),
        )
        self.assertNotIn(UP, state.legal_pawn_actions())

    def test_straight_jump_over_adjacent_opponent(self):
        state = GameState(size=5, pawns=((3, 2), (2, 2)), walls_remaining=(2, 2))
        self.assertIn(UP, state.legal_pawn_actions())
        self.assertEqual(state.apply_action(UP).pawns[0], (1, 2))
        self.assertNotIn(UP_LEFT, state.legal_pawn_actions())
        self.assertNotIn(UP_RIGHT, state.legal_pawn_actions())

    def test_diagonal_go_around_when_straight_jump_is_blocked(self):
        state = GameState(
            size=5,
            pawns=((3, 2), (2, 2)),
            horizontal_walls=frozenset({(1, 2)}),
            walls_remaining=(2, 2),
        )
        actions = state.legal_pawn_actions()
        self.assertIn(UP_LEFT, actions)
        self.assertIn(UP_RIGHT, actions)
        self.assertNotIn(UP, actions)
        self.assertEqual(state.apply_action(UP_LEFT).pawns[0], (2, 1))

    def test_diagonal_go_around_at_board_edge(self):
        state = GameState(size=5, pawns=((1, 2), (0, 2)), walls_remaining=(2, 2))
        self.assertIn(UP_LEFT, state.legal_pawn_actions())
        self.assertIn(UP_RIGHT, state.legal_pawn_actions())
        self.assertNotIn(UP, state.legal_pawn_actions())


class WallLegalityTests(unittest.TestCase):
    def test_fast_candidate_path_check_matches_materialized_state(self):
        rng = random.Random(91)
        state = GameState.initial(size=5, walls_per_player=4)
        for _ in range(12):
            legal_walls = [action for action in state.legal_actions() if action >= 8]
            if not legal_walls:
                break
            action = rng.choice(legal_walls)
            orientation, row, col = decode_wall_action(action, state.size)
            candidate = state._with_wall(orientation, row, col)
            for player in (0, 1):
                self.assertEqual(
                    state._has_path_with_extra_wall(player, orientation, row, col),
                    candidate.has_path(player),
                )
            state = state.apply_action(action)

    def test_wall_decrements_inventory_and_blocks_edges(self):
        state = GameState.initial(size=5, walls_per_player=2)
        action = encode_wall_action("H", 2, 1, 5)
        placed = state.apply_action(action)
        self.assertIn((2, 1), placed.horizontal_walls)
        self.assertEqual(placed.walls_remaining, (1, 2))
        self.assertTrue(placed.edge_blocked((2, 1), (3, 1)))
        self.assertTrue(placed.edge_blocked((2, 2), (3, 2)))

    def test_overlapping_and_crossing_walls_are_illegal(self):
        state = GameState(
            size=5,
            pawns=((4, 2), (0, 2)),
            horizontal_walls=frozenset({(2, 1)}),
            walls_remaining=(2, 2),
        )
        illegal = {
            encode_wall_action("H", 2, 0, 5),
            encode_wall_action("H", 2, 1, 5),
            encode_wall_action("H", 2, 2, 5),
            encode_wall_action("V", 2, 1, 5),
        }
        self.assertTrue(illegal.isdisjoint(state.legal_actions()))

    def test_wall_may_not_remove_last_path(self):
        state = GameState(
            size=5,
            pawns=((4, 2), (0, 2)),
            horizontal_walls=frozenset({(0, 0), (0, 2)}),
            walls_remaining=(2, 2),
        )
        closing = encode_wall_action("V", 0, 3, 5)
        self.assertNotIn(closing, state.legal_actions())
        with self.assertRaises(IllegalAction):
            state.apply_action(closing)


class TerminalAndCanonicalTests(unittest.TestCase):
    def test_reaching_any_goal_square_wins(self):
        for col in range(5):
            state = GameState(size=5, pawns=((1, col), (4, 4)), walls_remaining=(0, 0))
            won = state.apply_action(UP)
            self.assertTrue(won.is_terminal())
            self.assertEqual(won.winner, 0)
            self.assertEqual(won.outcome_for_current_player(), -1.0)

    def test_canonical_view_rotates_player_two_to_bottom(self):
        state = GameState(
            size=5,
            pawns=((3, 1), (1, 3)),
            horizontal_walls=frozenset({(0, 1)}),
            vertical_walls=frozenset({(2, 0)}),
            walls_remaining=(1, 2),
            turn=1,
        )
        canonical = state.canonical()
        self.assertEqual(canonical.turn, 0)
        self.assertEqual(canonical.pawns, ((3, 1), (1, 3)))
        self.assertEqual(canonical.walls_remaining, (2, 1))
        self.assertIn((3, 2), canonical.horizontal_walls)
        self.assertIn((1, 3), canonical.vertical_walls)

    def test_serialization_round_trip_and_stable_key(self):
        state = GameState(
            size=5,
            pawns=((3, 1), (1, 3)),
            horizontal_walls=frozenset({(0, 1)}),
            vertical_walls=frozenset({(2, 0)}),
            walls_remaining=(1, 2),
            turn=1,
        )
        restored = GameState.from_dict(json.loads(json.dumps(state.to_dict())))
        self.assertEqual(restored, state)
        self.assertEqual(restored.canonical_key(), state.canonical_key())


class RandomGameInvariantTests(unittest.TestCase):
    def test_random_games_remain_valid_and_terminate(self):
        rng = random.Random(7)
        for _ in range(100):
            state = GameState.initial(size=5, walls_per_player=2)
            for _ply in range(1000):
                self.assertTrue(state.has_path(0))
                self.assertTrue(state.has_path(1))
                if state.is_terminal():
                    break
                actions = state.legal_actions()
                self.assertTrue(actions)
                pawn_actions = state.legal_pawn_actions()
                # Bias the invariant test toward progress while retaining random walls.
                action = rng.choice(pawn_actions if rng.random() < 0.8 else actions)
                state = state.apply_action(action)
            else:
                self.fail("random game did not terminate within 1000 plies")


if __name__ == "__main__":
    unittest.main()
