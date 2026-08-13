import json
import tempfile
import unittest
from pathlib import Path


class TrainingRunTests(unittest.TestCase):
    def test_training_cli_exposes_production_hyperparameters(self):
        from scripts.train_generations import build_parser, config_from_args

        args = build_parser().parse_args(
            [
                "--batch-size", "128",
                "--replay-capacity", "50000",
                "--learning-rate", "0.0001",
                "--max-plies", "750",
                "--promotion-score", "0.6",
                "--resume-latest",
            ]
        )
        config = config_from_args(args)

        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.replay_capacity, 50_000)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.max_plies, 750)
        self.assertEqual(config.promotion_score, 0.6)
        self.assertTrue(args.resume_latest)

    def test_run_ledger_persists_config_and_generation_summaries(self):
        from training.run_ledger import RunLedger

        with tempfile.TemporaryDirectory() as directory:
            ledger = RunLedger(directory, {"board_size": 9, "seed": 7})
            ledger.record({"generation": 1, "checkpoint": "generation_001.pt"})
            ledger.record({"generation": 2, "checkpoint": "generation_002.pt"})

            manifest = json.loads((Path(directory) / "run.json").read_text())
            records = [
                json.loads(line)
                for line in (Path(directory) / "generations.jsonl").read_text().splitlines()
            ]

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["config"], {"board_size": 9, "seed": 7})
        self.assertEqual([record["generation"] for record in records], [1, 2])
        self.assertTrue(all("recorded_at" in record for record in records))

    def test_run_ledger_rejects_config_changes_in_existing_directory(self):
        from training.run_ledger import RunLedger

        with tempfile.TemporaryDirectory() as directory:
            RunLedger(directory, {"board_size": 9, "seed": 7})
            with self.assertRaisesRegex(ValueError, "configuration does not match"):
                RunLedger(directory, {"board_size": 9, "seed": 8})

    def test_latest_checkpoint_uses_generation_number(self):
        from training.run_ledger import latest_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("generation_002.pt", "generation_010.pt", "generation_001.pt"):
                (root / name).touch()
            (root / "unrelated.pt").touch()

            latest = latest_checkpoint(root)

        self.assertEqual(latest.name, "generation_010.pt")

    def test_latest_checkpoint_rejects_empty_directory(self):
        from training.run_ledger import latest_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "no generation checkpoints"):
                latest_checkpoint(directory)


if __name__ == "__main__":
    unittest.main()
