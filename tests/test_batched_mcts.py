import random
import unittest

try:
    import torch
except ImportError:
    torch = None

from barricade.state import GameState


@unittest.skipIf(torch is None, "PyTorch is not installed")
class BatchedMCTSTests(unittest.TestCase):
    def test_multiple_roots_share_model_forward_calls(self):
        from barricade.batched_mcts import BatchedMCTS
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        torch.manual_seed(2)
        evaluator = NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1))
        states = [GameState.initial(size=5, walls_per_player=0) for _ in range(4)]
        results = BatchedMCTS(evaluator, simulations=8, rng=random.Random(3)).search_batch(states)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(sum(result.visits) == 8 for result in results))
        self.assertLess(evaluator.forward_calls, evaluator.positions_evaluated)
        self.assertGreater(evaluator.average_batch_size, 1.0)

    def test_batched_search_finds_immediate_wins(self):
        from barricade.actions import UP
        from barricade.batched_mcts import BatchedMCTS
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        states = [GameState(size=5, pawns=((1, col), (4, 4)), walls_remaining=(0, 0)) for col in range(4)]
        evaluator = NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1))
        results = BatchedMCTS(evaluator, simulations=24).search_batch(states, temperature=0)
        self.assertTrue(all(result.best_action == UP for result in results))


if __name__ == "__main__":
    unittest.main()