#!/usr/bin/env python3
import argparse
import json

from training.generations import GenerationConfig, GenerationTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--arena-games", type=int, default=2)
    parser.add_argument("--walls", type=int, default=0)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--checkpoint-dir", default="checkpoints/generations")
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()
    config = GenerationConfig(
        walls_per_player=args.walls,
        channels=args.channels,
        residual_blocks=args.blocks,
        self_play_games=args.games,
        simulations=args.simulations,
        training_steps=args.steps,
        arena_games=args.arena_games,
        batch_size=16,
        seed=args.seed,
    )
    trainer = GenerationTrainer(config, args.checkpoint_dir)
    if args.resume:
        trainer.resume(args.resume)
    for _ in range(args.generations):
        print(json.dumps(trainer.run_generation(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()