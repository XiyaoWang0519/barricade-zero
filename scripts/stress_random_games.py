#!/usr/bin/env python3
import argparse
import random
import time

from barricade.state import GameState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--walls", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-plies", type=int, default=1000)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    started = time.perf_counter()
    total_plies = 0
    wins = [0, 0]
    for game in range(args.games):
        state = GameState.initial(args.size, args.walls)
        for ply in range(1, args.max_plies + 1):
            assert state.has_path(0) and state.has_path(1)
            actions = state.legal_actions()
            pawn_actions = state.legal_pawn_actions()
            action = rng.choice(pawn_actions if rng.random() < 0.8 else actions)
            state = state.apply_action(action)
            if state.is_terminal():
                wins[state.winner] += 1
                total_plies += ply
                break
        else:
            raise RuntimeError(f"game {game} exceeded {args.max_plies} plies")
    elapsed = time.perf_counter() - started
    print(f"games={args.games} wins={wins} plies={total_plies} seconds={elapsed:.3f} games_per_second={args.games/elapsed:.1f}")


if __name__ == "__main__":
    main()
