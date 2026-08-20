#!/usr/bin/env python3
"""Generate fixed openings and evaluate checkpoints without touching training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random

import torch

from barricade.agents import RandomAgent, ShortestPathAgent
from barricade.evaluation.arena import play_paired_match
from barricade.evaluation.checkpoints import (
    checkpoint_player_factory,
    untrained_player_factory,
)
from barricade.evaluation.openings import (
    generate_opening_suite,
    load_opening_suite,
    save_opening_suite,
)
from barricade.evaluation.players import AgentPlayerFactory, uniform_mcts_factory
from barricade.evaluation.predictions import (
    evaluate_predictions,
    load_labeled_positions,
)
from barricade.evaluation.reports import build_scoreboard, load_reports


def _write_json(path: str | Path, data: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def _validate_checkpoint(info, suite, role: str) -> None:
    if info.spec.board_size != suite.board_size:
        raise ValueError(
            f"{role} board size {info.spec.board_size} does not match "
            f"opening suite size {suite.board_size}"
        )
    if (
        info.walls_per_player is not None
        and info.walls_per_player != suite.walls_per_player
    ):
        raise ValueError(
            f"{role} wall count {info.walls_per_player} does not match "
            f"opening suite count {suite.walls_per_player}"
        )


def _generate_openings(args) -> None:
    suite = generate_opening_suite(
        count=args.count,
        board_size=args.board_size,
        walls_per_player=args.walls,
        min_plies=args.min_plies,
        max_plies=args.max_plies,
        seed=args.seed,
    )
    save_opening_suite(args.output, suite)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "openings": len(suite.openings),
                "sha256": suite.sha256,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _baseline_factory(name, args, candidate_info):
    if name == "random":
        return AgentPlayerFactory(
            "random",
            lambda seed: RandomAgent(random.Random(seed)),
        )
    if name == "shortest-path":
        return AgentPlayerFactory(
            "shortest_path",
            lambda _seed: ShortestPathAgent(),
        )
    if name == "uniform-mcts":
        return uniform_mcts_factory(args.simulations)
    if name == "untrained":
        return untrained_player_factory(
            candidate_info.spec,
            simulations=args.simulations,
            seed=args.untrained_seed,
            device=args.device,
            mixed_precision=args.mixed_precision,
        )
    raise ValueError(f"unknown baseline: {name}")


def _run_matches(args) -> None:
    suite = load_opening_suite(args.openings)
    candidate, candidate_info = checkpoint_player_factory(
        args.candidate,
        simulations=args.simulations,
        device=args.device,
        mixed_precision=args.mixed_precision,
        label="candidate",
    )
    _validate_checkpoint(candidate_info, suite, "candidate")
    opponents = []
    opponent_metadata = []
    champion_name = None
    champion_info = None
    if args.champion:
        champion, info = checkpoint_player_factory(
            args.champion,
            simulations=args.simulations,
            device=args.device,
            mixed_precision=args.mixed_precision,
            label="champion",
        )
        _validate_checkpoint(info, suite, "champion")
        opponents.append(champion)
        opponent_metadata.append({"label": champion.name, **info.to_dict()})
        champion_name = champion.name
        champion_info = info
    for index, path in enumerate(args.opponent_checkpoint):
        opponent, info = checkpoint_player_factory(
            path,
            simulations=args.simulations,
            device=args.device,
            mixed_precision=args.mixed_precision,
            label=f"checkpoint_{index}:{Path(path).stem}",
        )
        _validate_checkpoint(info, suite, "opponent")
        opponents.append(opponent)
        opponent_metadata.append({"label": opponent.name, **info.to_dict()})
    opponents.extend(
        _baseline_factory(name, args, candidate_info) for name in args.baseline
    )
    if not opponents:
        raise ValueError("select a champion, opponent checkpoint, or baseline")

    matches = []
    for index, opponent in enumerate(opponents):
        result = play_paired_match(
            candidate,
            opponent,
            suite,
            max_plies=args.max_plies,
            seed=args.seed + index,
            confidence=args.confidence,
            bootstrap_resamples=args.bootstrap_resamples,
            minimum_pairs=args.minimum_pairs,
            threshold=args.promotion_threshold,
        )
        matches.append(result)
        print(
            json.dumps(
                {
                    "opponent": result.opponent,
                    "score": result.score,
                    "wins": result.wins,
                    "draws": result.draws,
                    "losses": result.losses,
                    "interval": [result.interval.lower, result.interval.upper],
                    "decision": result.decision.status,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    champion_match = next(
        (result for result in matches if result.opponent == champion_name),
        None,
    )
    if (
        champion_match is not None
        and champion_info is not None
        and candidate_info.model_sha256 == champion_info.model_sha256
    ):
        promotion = {
            "status": "identical_model",
            "eligible": False,
            "reason": "candidate and champion have identical model weights",
        }
    elif champion_match is not None:
        promotion = champion_match.decision.to_dict()
    else:
        promotion = {
            "status": "not_evaluated",
            "eligible": False,
            "reason": "no champion checkpoint was supplied",
        }
    report = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": candidate_info.to_dict(),
        "opponent_checkpoints": opponent_metadata,
        "opening_suite": {
            "path": str(Path(args.openings).resolve()),
            "sha256": suite.sha256,
            "board_size": suite.board_size,
            "walls_per_player": suite.walls_per_player,
            "pairs": len(suite.openings),
        },
        "settings": {
            "simulations": args.simulations,
            "max_plies": args.max_plies,
            "seed": args.seed,
            "confidence": args.confidence,
            "bootstrap_resamples": args.bootstrap_resamples,
            "minimum_pairs": args.minimum_pairs,
            "promotion_threshold": args.promotion_threshold,
            "device": args.device,
            "mixed_precision": args.mixed_precision,
            "torch_threads": args.torch_threads,
        },
        "matches": [result.to_dict() for result in matches],
        "promotion": promotion,
    }
    _write_json(args.output, report)
    print(json.dumps({"report": str(Path(args.output).resolve())}), flush=True)


def _run_predictions(args) -> None:
    player_factory, checkpoint_info = checkpoint_player_factory(
        args.candidate,
        simulations=1,
        device=args.device,
        mixed_precision=args.mixed_precision,
        label="candidate",
    )
    positions = load_labeled_positions(args.dataset)
    if any(item.state.size != checkpoint_info.spec.board_size for item in positions):
        raise ValueError("dataset board size does not match checkpoint")
    metrics = evaluate_predictions(
        player_factory.evaluator,
        positions,
        batch_size=args.batch_size,
    )
    report = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": checkpoint_info.to_dict(),
        "dataset": str(Path(args.dataset).resolve()),
        "settings": {
            "batch_size": args.batch_size,
            "device": args.device,
            "mixed_precision": args.mixed_precision,
            "torch_threads": args.torch_threads,
        },
        "metrics": metrics.to_dict(),
    }
    _write_json(args.output, report)
    print(json.dumps(report["metrics"], sort_keys=True), flush=True)


def _build_scoreboard(args) -> None:
    scoreboard = build_scoreboard(load_reports(args.report))
    _write_json(args.output, scoreboard)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "generations": len(scoreboard["generations"]),
                "comparable": scoreboard["comparable"],
                "warnings": scoreboard["warnings"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible Barricade checkpoint evaluation"
    )
    parser.add_argument("--torch-threads", type=int)
    subparsers = parser.add_subparsers(dest="command", required=True)

    openings = subparsers.add_parser(
        "generate-openings", help="create a versioned fixed opening suite"
    )
    openings.add_argument("--output", required=True)
    openings.add_argument("--count", type=int, default=200)
    openings.add_argument("--board-size", type=int, default=9)
    openings.add_argument("--walls", type=int, default=10)
    openings.add_argument("--min-plies", type=int, default=2)
    openings.add_argument("--max-plies", type=int, default=16)
    openings.add_argument("--seed", type=int, default=20260813)
    openings.set_defaults(handler=_generate_openings)

    run = subparsers.add_parser("run", help="run paired matches and write a report")
    run.add_argument("--candidate", required=True)
    run.add_argument("--openings", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--champion")
    run.add_argument("--opponent-checkpoint", action="append", default=[])
    run.add_argument(
        "--baseline",
        action="append",
        choices=("random", "shortest-path", "uniform-mcts", "untrained"),
        default=[],
    )
    run.add_argument("--simulations", type=int, default=64)
    run.add_argument("--max-plies", type=int, default=500)
    run.add_argument("--seed", type=int, default=20260813)
    run.add_argument("--confidence", type=float, default=0.95)
    run.add_argument("--bootstrap-resamples", type=int, default=10_000)
    run.add_argument("--minimum-pairs", type=int, default=100)
    run.add_argument("--promotion-threshold", type=float, default=0.5)
    run.add_argument("--untrained-seed", type=int, default=1)
    run.add_argument("--device", default="cpu")
    run.add_argument("--mixed-precision", action="store_true")
    run.set_defaults(handler=_run_matches)

    predictions = subparsers.add_parser(
        "predictions", help="score a checkpoint on held-out labeled positions"
    )
    predictions.add_argument("--candidate", required=True)
    predictions.add_argument("--dataset", required=True)
    predictions.add_argument("--output", required=True)
    predictions.add_argument("--batch-size", type=int, default=256)
    predictions.add_argument("--device", default="cpu")
    predictions.add_argument("--mixed-precision", action="store_true")
    predictions.set_defaults(handler=_run_predictions)

    scoreboard = subparsers.add_parser(
        "scoreboard", help="combine reports into a comparable generation history"
    )
    scoreboard.add_argument("--report", action="append", required=True)
    scoreboard.add_argument("--output", required=True)
    scoreboard.set_defaults(handler=_build_scoreboard)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.torch_threads is not None:
        if args.torch_threads <= 0:
            raise ValueError("torch_threads must be positive")
        torch.set_num_threads(args.torch_threads)
    args.handler(args)


if __name__ == "__main__":
    main()
