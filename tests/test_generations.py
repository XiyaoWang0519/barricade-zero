import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class GenerationDeviceTests(unittest.TestCase):
    def test_self_play_forwards_device_and_mixed_precision_to_evaluator(self):
        from training.generations import GenerationConfig, GenerationTrainer

        calls = []

        class RecordingEvaluator:
            def __init__(self, model, device, mixed_precision=False):
                calls.append((str(device), mixed_precision))

        result = SimpleNamespace(
            examples=[],
            game_lengths=[],
            forward_calls=0,
            positions_evaluated=0,
            average_inference_batch_size=0.0,
            draws=0,
            reused_root_visits=0,
        )
        config = GenerationConfig(
            channels=8,
            residual_blocks=1,
            self_play_games=1,
            simulations=1,
            mixed_precision=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            trainer = GenerationTrainer(config, directory, device="cpu")
            with (
                patch("training.generations.NeuralEvaluator", RecordingEvaluator),
                patch("training.generations.play_concurrent_games", return_value=result),
            ):
                trainer._self_play()

        self.assertEqual(calls, [("cpu", True)])

    def test_generation_reports_parameter_and_checkpoint_hash_changes(self):
        from training.generations import GenerationConfig, GenerationTrainer

        config = GenerationConfig(
            board_size=5,
            walls_per_player=0,
            channels=8,
            residual_blocks=1,
            self_play_games=2,
            simulations=1,
            training_steps=1,
            batch_size=4,
            arena_games=2,
            max_plies=50,
            seed=404,
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = GenerationTrainer(config, directory).run_generation()

        self.assertNotEqual(
            summary["initial_candidate_sha256"],
            summary["trained_candidate_sha256"],
        )
        self.assertEqual(len(summary["checkpoint_sha256"]), 64)
        self.assertGreaterEqual(summary["self_play_seconds"], 0.0)
        self.assertGreaterEqual(summary["candidate_setup_seconds"], 0.0)
        self.assertGreaterEqual(summary["training_seconds"], 0.0)
        self.assertGreaterEqual(summary["candidate_hash_seconds"], 0.0)
        self.assertGreaterEqual(summary["arena_seconds"], 0.0)
        self.assertGreaterEqual(summary["promotion_seconds"], 0.0)
        self.assertGreaterEqual(summary["checkpoint_seconds"], 0.0)
        self.assertGreaterEqual(summary["generation_seconds"], 0.0)
        self.assertAlmostEqual(
            summary["generation_seconds"],
            summary["self_play_seconds"]
            + summary["candidate_setup_seconds"]
            + summary["training_seconds"]
            + summary["candidate_hash_seconds"]
            + summary["arena_seconds"]
            + summary["promotion_seconds"]
            + summary["checkpoint_seconds"]
            + summary["generation_overhead_seconds"],
            places=6,
        )

    def test_resume_restores_replay_and_rng_state(self):
        from training.generations import GenerationConfig, GenerationTrainer

        config = GenerationConfig(
            board_size=5,
            walls_per_player=0,
            channels=8,
            residual_blocks=1,
            self_play_games=2,
            simulations=1,
            training_steps=1,
            batch_size=4,
            arena_games=2,
            max_plies=50,
            seed=405,
        )
        with tempfile.TemporaryDirectory() as directory:
            original = GenerationTrainer(config, directory)
            summary = original.run_generation()
            expected_replay = list(original.replay)
            expected_rng_state = original.rng.getstate()

            resumed = GenerationTrainer(config, directory)
            resumed.resume(summary["checkpoint"])

        self.assertEqual(resumed.generation, 1)
        self.assertEqual(list(resumed.replay), expected_replay)
        self.assertEqual(resumed.rng.getstate(), expected_rng_state)


if __name__ == "__main__":
    unittest.main()
