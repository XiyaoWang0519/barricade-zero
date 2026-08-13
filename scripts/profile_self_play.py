#!/usr/bin/env python3
"""Reproducible cProfile harness for rules, MCTS, and batched inference."""

import argparse
import cProfile
import io
import json
import pstats
import random
import time
from pathlib import Path

import torch

from barricade.concurrent_self_play import play_concurrent_games
from barricade.state import GameState
from neural.evaluator import NeuralEvaluator
from neural.model import PolicyValueNetwork


def run_workload(board_size: int, walls: int, games: int, simulations: int, seed: int) -> dict:
    torch.manual_seed(seed)
    evaluator = NeuralEvaluator(
        PolicyValueNetwork(board_size, channels=16, residual_blocks=2)
    )
    started = time.perf_counter()
    result = play_concurrent_games(
        evaluator,
        games=games,
        simulations=simulations,
        board_size=board_size,
        walls_per_player=walls,
        rng=random.Random(seed),
        max_plies=300,
    )
    elapsed = time.perf_counter() - started
    return {
        "board_size": board_size,
        "walls": walls,
        "games": games,
        "simulations": simulations,
        "examples": len(result.examples),
        "draws": result.draws,
        "seconds": elapsed,
        "positions": result.positions_evaluated,
        "forward_calls": result.forward_calls,
        "average_batch_size": result.average_inference_batch_size,
        "positions_per_second": result.positions_evaluated / elapsed,
        "reused_root_visits": result.reused_root_visits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--walls", type=int, default=10)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--profile", default="profiles/self_play.prof")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    profile_path = Path(args.profile)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    profiler.enable()
    summary = run_workload(
        args.board_size, args.walls, args.games, args.simulations, args.seed
    )
    profiler.disable()
    profiler.dump_stats(profile_path)
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative")
    stats.print_stats(args.top)
    summary["profile"] = str(profile_path)
    print(json.dumps(summary, sort_keys=True))
    print(stream.getvalue())


if __name__ == "__main__":
    main()
