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


if __name__ == "__main__":
    unittest.main()