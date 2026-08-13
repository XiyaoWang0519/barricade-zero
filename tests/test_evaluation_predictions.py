import unittest

from barricade.evaluation.predictions import LabeledPosition, evaluate_predictions
from barricade.state import GameState


class _FixedEvaluator:
    def __init__(self, policy, value):
        self.policy = policy
        self.value = value

    def evaluate_batch(self, states):
        return [(self.policy, self.value) for _state in states]


class EvaluationPredictionTests(unittest.TestCase):
    def test_prediction_metrics_use_unseen_policy_and_value_targets(self):
        state = GameState.initial(size=5, walls_per_player=0)
        legal = state.legal_actions()
        policy = [0.0] * state.action_size
        policy[legal[0]] = 0.75
        policy[legal[1]] = 0.25
        positions = [
            LabeledPosition("a", state, policy, 1.0),
            LabeledPosition("b", state, policy, 0.0),
        ]
        result = evaluate_predictions(
            _FixedEvaluator(policy, 0.5),
            positions,
            batch_size=2,
        )
        self.assertEqual(result.positions, 2)
        self.assertEqual(result.policy_top1_accuracy, 1.0)
        self.assertEqual(result.policy_top3_accuracy, 1.0)
        self.assertAlmostEqual(result.value_mae, 0.5)
        self.assertAlmostEqual(result.value_mse, 0.25)
        self.assertAlmostEqual(result.value_calibration_error, 0.0)
        self.assertAlmostEqual(result.illegal_policy_mass, 0.0)

    def test_labeled_position_rejects_invalid_policy(self):
        state = GameState.initial(size=5, walls_per_player=0)
        with self.assertRaises(ValueError):
            LabeledPosition("bad", state, [1.0], 1.0)


if __name__ == "__main__":
    unittest.main()
