#!/usr/bin/env python3
import argparse
import json
import random
import time

import torch

from barricade.concurrent_self_play import play_concurrent_games
from neural.evaluator import NeuralEvaluator
from neural.model import PolicyValueNetwork


def run(model, games: int, simulations: int, seed: int, reuse: bool) -> dict:
    evaluator = NeuralEvaluator(model)
    started = time.perf_counter()
    result = play_concurrent_games(
        evaluator,
        games=games,
        simulations=simulations,
        board_size=5,
        walls_per_player=0,
        rng=random.Random(seed),
        max_plies=200,
        use_tree_reuse=reuse,
    )
    seconds = time.perf_counter() - started
    return {
        "examples": len(result.examples),
        "seconds": seconds,
        "forward_calls": result.forward_calls,
        "positions": result.positions_evaluated,
        "average_batch_size": result.average_inference_batch_size,
        "reused_root_visits": result.reused_root_visits,
        "positions_per_example": result.positions_evaluated / len(result.examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model = PolicyValueNetwork(5, channels=16, residual_blocks=2)
    without = run(model, args.games, args.simulations, args.seed, False)
    with_reuse = run(model, args.games, args.simulations, args.seed, True)
    print(json.dumps({
        "games": args.games,
        "simulations": args.simulations,
        "without_reuse": {key: round(value, 4) if isinstance(value, float) else value for key, value in without.items()},
        "with_reuse": {key: round(value, 4) if isinstance(value, float) else value for key, value in with_reuse.items()},
        "positions_per_example_reduction": round(
            without["positions_per_example"] / with_reuse["positions_per_example"], 3
        ),
        "wall_clock_speedup": round(without["seconds"] / with_reuse["seconds"], 3),
    }, sort_keys=True))


if __name__ == "__main__":
    main()