import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_restores_model_optimizer_and_metadata(self):
        from neural.model import PolicyValueNetwork
        from training.checkpoint import load_checkpoint, save_checkpoint
        from training.learner import Learner

        torch.manual_seed(12)
        model = PolicyValueNetwork(5, channels=8, residual_blocks=1)
        learner = Learner(model)
        expected = [parameter.detach().clone() for parameter in model.parameters()]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation_003.pt"
            save_checkpoint(path, model, learner.optimizer, generation=3, metadata={"score": 0.75})
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            payload = load_checkpoint(path, model, learner.optimizer)
        self.assertEqual(payload["generation"], 3)
        self.assertEqual(payload["metadata"]["score"], 0.75)
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(expected, model.parameters())))


if __name__ == "__main__":
    unittest.main()