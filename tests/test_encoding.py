import unittest

from barricade.actions import encode_wall_action, rotate_action
from barricade.encoding import encode_state, legal_action_mask, rotate_policy
from barricade.state import GameState


class NeuralEncodingTests(unittest.TestCase):
    def test_state_encoding_has_eight_planes(self):
        state = GameState.initial(size=5, walls_per_player=2)
        planes = encode_state(state)
        self.assertEqual((len(planes), len(planes[0]), len(planes[0][0])), (8, 5, 5))
        self.assertEqual(planes[0][4][2], 1.0)
        self.assertEqual(planes[1][0][2], 1.0)
        self.assertTrue(all(value == 1.0 for row in planes[4] for value in row))

    def test_encoding_uses_canonical_side_to_move_view(self):
        state = GameState(size=5, pawns=((3, 1), (1, 3)), walls_remaining=(1, 2), turn=1)
        planes = encode_state(state, max_walls=2)
        self.assertEqual(planes[0][3][1], 1.0)
        self.assertEqual(planes[1][1][3], 1.0)
        self.assertEqual(planes[4][0][0], 1.0)
        self.assertEqual(planes[5][0][0], 0.5)

    def test_wall_anchor_planes_and_legal_mask(self):
        state = GameState(
            size=5,
            pawns=((4, 2), (0, 2)),
            horizontal_walls=frozenset({(2, 1)}),
            walls_remaining=(2, 2),
        )
        planes = encode_state(state, max_walls=2)
        self.assertEqual(planes[2][2][1], 1.0)
        mask = legal_action_mask(state)
        self.assertEqual(len(mask), 40)
        self.assertFalse(mask[encode_wall_action("H", 2, 1, 5)])
        self.assertEqual(sum(mask), len(state.legal_actions()))

    def test_policy_rotation_round_trip(self):
        policy = [float(index) for index in range(136)]
        rotated = rotate_policy(policy, 9)
        restored = rotate_policy(rotated, 9)
        self.assertEqual(restored, policy)
        for action, value in enumerate(policy):
            self.assertEqual(rotated[rotate_action(action, 9)], value)


if __name__ == "__main__":
    unittest.main()