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


@unittest.skipUnless(
    torch is not None and torch.cuda.is_available(), "CUDA is not available"
)
class CudaCheckpointTests(unittest.TestCase):
    def test_checkpoint_loads_cpu_to_cuda_and_back_to_cpu(self):
        from neural.model import PolicyValueNetwork
        from training.checkpoint import load_checkpoint, save_checkpoint
        from training.learner import Learner

        torch.manual_seed(402)
        cpu_model = PolicyValueNetwork(5, channels=8, residual_blocks=1)
        cpu_learner = Learner(cpu_model)
        for parameter in cpu_model.parameters():
            parameter.grad = torch.ones_like(parameter)
        cpu_learner.optimizer.step()
        expected = [parameter.detach().clone() for parameter in cpu_model.parameters()]
        with tempfile.TemporaryDirectory() as directory:
            cpu_path = Path(directory) / "cpu.pt"
            cuda_path = Path(directory) / "cuda.pt"
            save_checkpoint(
                cpu_path, cpu_model, cpu_learner.optimizer, generation=1
            )

            cuda_model = PolicyValueNetwork(5, channels=8, residual_blocks=1).to("cuda")
            cuda_learner = Learner(cuda_model, device="cuda")
            load_checkpoint(
                cpu_path, cuda_model, cuda_learner.optimizer, map_location="cuda"
            )
            self.assertTrue(all(parameter.is_cuda for parameter in cuda_model.parameters()))
            self.assertTrue(
                all(
                    state[name].is_cuda
                    for state in cuda_learner.optimizer.state.values()
                    for name in ("exp_avg", "exp_avg_sq")
                )
            )
            save_checkpoint(
                cuda_path, cuda_model, cuda_learner.optimizer, generation=2
            )

            restored = PolicyValueNetwork(5, channels=8, residual_blocks=1)
            restored_learner = Learner(restored)
            payload = load_checkpoint(
                cuda_path, restored, restored_learner.optimizer, map_location="cpu"
            )
        self.assertEqual(payload["generation"], 2)
        self.assertTrue(
            all(
                not state[name].is_cuda
                for state in restored_learner.optimizer.state.values()
                for name in ("exp_avg", "exp_avg_sq")
            )
        )
        self.assertTrue(
            all(
                torch.equal(old, new)
                for old, new in zip(expected, restored.parameters())
            )
        )


if __name__ == "__main__":
    unittest.main()
