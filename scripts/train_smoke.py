#!/usr/bin/env python3
"""Run a tiny end-to-end 5x5 AlphaZero smoke cycle on CPU."""

import argparse
import json
import random
import time
from pathlib import Path

import torch

from barricade.mcts import MCTS, UniformEvaluator
from barricade.self_play import play_self_play_game
from barricade.state import GameState
from neural.model import PolicyValueNetwork
from training.learner import Learner, ReplayBuffer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=12)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", default="checkpoints/smoke.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    replay = ReplayBuffer(1000, random.Random(args.seed))
    search = MCTS(
        UniformEvaluator(), simulations=args.simulations, rng=random.Random(args.seed + 1)
    )
    started = time.perf_counter()
    game_lengths = []
    for game in range(args.games):
        examples = play_self_play_game(
            GameState.initial(size=5, walls_per_player=0),
            search,
            rng=random.Random(args.seed + game + 2),
            exploration_plies=4,
        )
        replay.extend(examples)
        game_lengths.append(len(examples))

    model = PolicyValueNetwork(board_size=5, channels=16, residual_blocks=2)
    learner = Learner(model, learning_rate=1e-3)
    metrics = []
    for _ in range(args.steps):
        metrics.append(learner.train_batch(replay.sample(min(args.batch_size, len(replay)))))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "board_size": 5, "examples": len(replay)}, output
    )
    summary = {
        "games": args.games,
        "examples": len(replay),
        "game_lengths": game_lengths,
        "steps": args.steps,
        "initial_loss": metrics[0]["loss"],
        "final_loss": metrics[-1]["loss"],
        "checkpoint": str(output),
        "seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()