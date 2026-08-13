import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class EvaluationCheckpointTests(unittest.TestCase):
    def test_checkpoint_loader_infers_network_and_preserves_metadata(self):
        from barricade.evaluation.checkpoints import load_checkpoint_evaluator
        from barricade.state import GameState
        from neural.model import PolicyValueNetwork
        from training.checkpoint import save_checkpoint
        from training.learner import Learner

        model = PolicyValueNetwork(5, channels=8, residual_blocks=1)
        learner = Learner(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation_007.pt"
            save_checkpoint(
                path,
                model,
                learner.optimizer,
                generation=7,
                metadata={"config": {"walls_per_player": 2}},
            )
            evaluator, info = load_checkpoint_evaluator(path)
        self.assertEqual(info.generation, 7)
        self.assertEqual(info.walls_per_player, 2)
        self.assertEqual(info.spec.board_size, 5)
        self.assertEqual(info.spec.channels, 8)
        self.assertEqual(info.spec.residual_blocks, 1)
        self.assertEqual(len(info.sha256), 64)
        self.assertEqual(len(info.model_sha256), 64)
        policy, value = evaluator.evaluate(GameState.initial(5, 2))
        self.assertEqual(len(policy), 40)
        self.assertTrue(-1.0 <= value <= 1.0)


if __name__ == "__main__":
    unittest.main()
