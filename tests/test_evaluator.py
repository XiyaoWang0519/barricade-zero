import unittest

try:
    import torch
except ImportError:
    torch = None

from barricade.actions import rotate_action
from barricade.state import GameState


@unittest.skipIf(torch is None, "PyTorch is not installed")
class NeuralEvaluatorTests(unittest.TestCase):
    def test_policy_is_legal_and_normalized_for_both_players(self):
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        torch.manual_seed(4)
        evaluator = NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1))
        first = GameState.initial(size=5, walls_per_player=2)
        second = first.apply_action(first.legal_pawn_actions()[0])
        for state in (first, second):
            policy, value = evaluator.evaluate(state)
            self.assertAlmostEqual(sum(policy), 1.0, places=6)
            self.assertTrue(-1 <= value <= 1)
            legal = set(state.legal_actions())
            self.assertTrue(all(policy[action] == 0 for action in range(state.action_size) if action not in legal))

    def test_player_two_policy_is_rotated_back_to_original_coordinates(self):
        from neural.evaluator import NeuralEvaluator

        class FixedModel(torch.nn.Module):
            def forward(self, inputs):
                logits = torch.arange(40, dtype=torch.float32).repeat(inputs.shape[0], 1)
                return logits, torch.zeros(inputs.shape[0], 1)

        state = GameState.initial(size=5, walls_per_player=0).apply_action(0)
        evaluator = NeuralEvaluator(FixedModel())
        policy, _ = evaluator.evaluate(state)
        canonical = state.canonical()
        canonical_policy, _ = NeuralEvaluator(FixedModel()).evaluate(canonical)
        for action in state.legal_actions():
            self.assertAlmostEqual(policy[action], canonical_policy[rotate_action(action, 5)])

    def test_batch_evaluation_matches_individual_evaluation(self):
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        torch.manual_seed(8)
        evaluator = NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1))
        first = GameState.initial(size=5, walls_per_player=0)
        second = first.apply_action(first.legal_pawn_actions()[0])
        batched = evaluator.evaluate_batch([first, second])
        individual = [evaluator.evaluate(first), evaluator.evaluate(second)]
        for (batch_policy, batch_value), (single_policy, single_value) in zip(batched, individual):
            self.assertEqual(len(batch_policy), len(single_policy))
            for actual, expected in zip(batch_policy, single_policy):
                self.assertAlmostEqual(actual, expected, places=6)
            self.assertAlmostEqual(batch_value, single_value, places=6)

    def test_batch_metrics_count_forward_calls_and_positions(self):
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        evaluator = NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1))
        states = [GameState.initial(size=5, walls_per_player=0)] * 3
        evaluator.evaluate_batch(states)
        evaluator.evaluate(states[0])
        self.assertEqual(evaluator.forward_calls, 2)
        self.assertEqual(evaluator.positions_evaluated, 4)
        self.assertEqual(evaluator.average_batch_size, 2.0)

    def test_unmasked_batch_skips_legal_action_generation(self):
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        state = GameState.initial(size=5, walls_per_player=2)
        evaluator = NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1))
        policy, value = evaluator.evaluate_batch([state], mask_legal=False)[0]
        self.assertAlmostEqual(sum(policy), 1.0, places=6)
        self.assertTrue(-1 <= value <= 1)


if __name__ == "__main__":
    unittest.main()