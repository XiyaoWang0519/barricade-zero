#!/usr/bin/env python3
import argparse
import json
import random
import time

import torch

from barricade.concurrent_self_play import play_concurrent_games
from barricade.mcts import MCTS
from barricade.self_play import play_self_play_game
from barricade.state import GameState
from neural.evaluator import NeuralEvaluator
from neural.model import PolicyValueNetwork


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model = PolicyValueNetwork(5, channels=16, residual_blocks=2)

    sequential_evaluator = NeuralEvaluator(model)
    started = time.perf_counter()
    sequential_examples = 0
    for game in range(args.games):
        examples = play_self_play_game(
            GameState.initial(5, 0),
            MCTS(
                sequential_evaluator,
                simulations=args.simulations,
                rng=random.Random(args.seed + game),
            ),
            rng=random.Random(args.seed + 100 + game),
            max_plies=200,
        )
        sequential_examples += len(examples)
    sequential_seconds = time.perf_counter() - started

    batched_evaluator = NeuralEvaluator(model)
    started = time.perf_counter()
    batched = play_concurrent_games(
        batched_evaluator,
        games=args.games,
        simulations=args.simulations,
        board_size=5,
        walls_per_player=0,
        rng=random.Random(args.seed + 200),
        max_plies=200,
    )
    batched_seconds = time.perf_counter() - started
    print(
        json.dumps(
            {
                "games": args.games,
                "simulations": args.simulations,
                "sequential_examples": sequential_examples,
                "sequential_forward_calls": sequential_evaluator.forward_calls,
                "sequential_seconds": round(sequential_seconds, 4),
                "batched_examples": len(batched.examples),
                "batched_forward_calls": batched.forward_calls,
                "batched_positions": batched.positions_evaluated,
                "average_batch_size": round(batched.average_inference_batch_size, 3),
                "batched_seconds": round(batched_seconds, 4),
                "forward_call_reduction": round(
                    sequential_evaluator.forward_calls / batched.forward_calls, 3
                ),
                "wall_clock_speedup": round(sequential_seconds / batched_seconds, 3),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()