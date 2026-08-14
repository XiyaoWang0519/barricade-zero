#!/usr/bin/env python3
"""Stable, phase-aware benchmark for one complete training generation."""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import platform
import pstats
import statistics
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from training.generations import GenerationConfig, GenerationTrainer


PHASE_METRICS = (
    "generation_seconds",
    "self_play_seconds",
    "candidate_setup_seconds",
    "training_seconds",
    "candidate_hash_seconds",
    "arena_seconds",
    "promotion_seconds",
    "checkpoint_seconds",
    "generation_overhead_seconds",
    "new_examples",
    "replay_examples",
    "positions_evaluated",
    "inference_forward_calls",
    "average_inference_batch_size",
    "examples_per_generation_second",
    "positions_per_generation_second",
)


def _distribution(values: list[float | int]) -> dict:
    median = statistics.median(values)
    if len(values) == 1:
        first_quartile = third_quartile = median
    else:
        first_quartile, _, third_quartile = statistics.quantiles(
            values, n=4, method="inclusive"
        )
    return {
        "count": len(values),
        "minimum": min(values),
        "median": median,
        "maximum": max(values),
        "population_stddev": statistics.pstdev(values),
        "median_absolute_deviation": statistics.median(
            [abs(value - median) for value in values]
        ),
        "first_quartile": first_quartile,
        "third_quartile": third_quartile,
        "interquartile_range": third_quartile - first_quartile,
    }


def summarize_runs(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("at least one measured run is required")
    aggregate = {"repetitions": len(runs)}
    for metric in PHASE_METRICS:
        values = [run[metric] for run in runs if metric in run]
        if values:
            aggregate[metric] = _distribution(values)
    return aggregate


def run_workload(config: GenerationConfig, device: str, checkpoint_dir: str | Path) -> dict:
    trainer = GenerationTrainer(config, checkpoint_dir, device=device)
    summary = trainer.run_generation()
    seconds = summary["generation_seconds"]
    summary["examples_per_generation_second"] = summary["new_examples"] / seconds
    summary["positions_per_generation_second"] = summary["positions_evaluated"] / seconds
    summary["torch_threads"] = torch.get_num_threads()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--walls", type=int, default=10)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=10_000)
    parser.add_argument("--arena-games", type=int, default=2)
    parser.add_argument("--promotion-score", type=float, default=0.55)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--profile")
    parser.add_argument("--json-output")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--top", type=int, default=25)
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
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        arena_games=args.arena_games,
        promotion_score=args.promotion_score,
        learning_rate=args.learning_rate,
        max_plies=args.max_plies,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
        torch_threads=args.torch_threads,
        cuda_graphs=args.cuda_graphs,
    )


def _environment_summary() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.arena_games <= 0:
        parser.error("--arena-games must be positive")
    config = config_from_args(args)

    with tempfile.TemporaryDirectory(prefix="barricade-generation-benchmark-") as root:
        runs = []
        for index in range(args.warmups):
            run_workload(config, args.device, Path(root) / f"warmup-{index + 1}")
            print(
                f"completed warmup {index + 1}/{args.warmups}",
                file=sys.stderr,
                flush=True,
            )
        for index in range(args.repetitions):
            runs.append(
                run_workload(config, args.device, Path(root) / f"run-{index + 1}")
            )
            print(
                f"completed measured repetition {index + 1}/{args.repetitions}",
                file=sys.stderr,
                flush=True,
            )

        report = {
            "environment": _environment_summary(),
            "config": asdict(config),
            "warmups": args.warmups,
            "runs": runs,
            "aggregate": summarize_runs(runs),
        }
        profile_text = None
        if args.profile:
            profile_path = Path(args.profile)
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profiler = cProfile.Profile()
            profiler.enable()
            profile_summary = run_workload(config, args.device, Path(root) / "profile")
            profiler.disable()
            profiler.dump_stats(profile_path)
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative")
            stats.print_stats(args.top)
            profile_summary["profile"] = str(profile_path)
            report["profile_run"] = profile_summary
            profile_text = stream.getvalue()

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not args.quiet:
        print(json.dumps(report, sort_keys=True))
    if profile_text is not None and not args.quiet:
        print(profile_text)


if __name__ == "__main__":
    main()
