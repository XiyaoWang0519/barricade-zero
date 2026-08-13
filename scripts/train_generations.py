#!/usr/bin/env python3
import argparse
import json
from dataclasses import asdict

from training.generations import GenerationConfig, GenerationTrainer
from training.run_ledger import RunLedger, latest_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--arena-games", type=int, default=2)
    parser.add_argument("--walls", type=int, default=0)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--replay-capacity", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-plies", type=int, default=500)
    parser.add_argument("--promotion-score", type=float, default=0.55)
    parser.add_argument("--checkpoint-dir", default="checkpoints/generations")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume")
    resume.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--seed", type=int, default=21)
    return parser


def config_from_args(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        board_size=args.board_size,
        walls_per_player=args.walls,
        channels=args.channels,
        residual_blocks=args.blocks,
        self_play_games=args.games,
        simulations=args.simulations,
        training_steps=args.steps,
        arena_games=args.arena_games,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        learning_rate=args.learning_rate,
        max_plies=args.max_plies,
        promotion_score=args.promotion_score,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
    )


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    ledger = RunLedger(args.checkpoint_dir, asdict(config))
    trainer = GenerationTrainer(config, args.checkpoint_dir, device=args.device)
    resume_path = latest_checkpoint(args.checkpoint_dir) if args.resume_latest else args.resume
    if resume_path:
        trainer.resume(resume_path)
    for _ in range(args.generations):
        summary = trainer.run_generation()
        ledger.record(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
