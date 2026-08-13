"""Small, resumable AlphaZero generation loop."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from barricade.arena import Arena, ArenaResult
from barricade.mcts import MCTS
from barricade.self_play import play_self_play_game
from barricade.state import GameState
from neural.evaluator import NeuralEvaluator
from neural.model import PolicyValueNetwork
from .checkpoint import load_checkpoint, save_checkpoint
from .learner import Learner, ReplayBuffer


@dataclass(frozen=True)
class GenerationConfig:
    board_size: int = 5
    walls_per_player: int = 2
    channels: int = 32
    residual_blocks: int = 3
    self_play_games: int = 8
    simulations: int = 25
    training_steps: int = 20
    batch_size: int = 64
    replay_capacity: int = 10_000
    arena_games: int = 10
    promotion_score: float = 0.55
    learning_rate: float = 3e-4
    max_plies: int = 500
    seed: int = 1


class GenerationTrainer:
    def __init__(self, config: GenerationConfig, checkpoint_dir: str | Path, device: str = "cpu") -> None:
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.rng = random.Random(config.seed)
        torch.manual_seed(config.seed)
        self.champion = self._new_model()
        self.generation = 0
        self.replay = ReplayBuffer(config.replay_capacity, self.rng)

    def _new_model(self) -> PolicyValueNetwork:
        return PolicyValueNetwork(
            self.config.board_size, self.config.channels, self.config.residual_blocks
        ).to(self.device)

    def resume(self, path: str | Path) -> None:
        payload = load_checkpoint(path, self.champion, map_location=self.device)
        self.generation = int(payload["generation"])

    def _self_play(self) -> tuple[int, list[int]]:
        evaluator = NeuralEvaluator(self.champion, self.device)
        game_lengths = []
        added = 0
        for _ in range(self.config.self_play_games):
            search = MCTS(
                evaluator,
                simulations=self.config.simulations,
                rng=random.Random(self.rng.getrandbits(64)),
            )
            examples = play_self_play_game(
                GameState.initial(self.config.board_size, self.config.walls_per_player),
                search,
                rng=random.Random(self.rng.getrandbits(64)),
                max_plies=self.config.max_plies,
            )
            self.replay.extend(examples)
            added += len(examples)
            game_lengths.append(len(examples))
        return added, game_lengths

    def run_generation(self) -> dict:
        added, game_lengths = self._self_play()
        candidate = copy.deepcopy(self.champion)
        learner = Learner(candidate, learning_rate=self.config.learning_rate, device=self.device)
        metrics = []
        for _ in range(self.config.training_steps):
            metrics.append(
                learner.train_batch(self.replay.sample(min(self.config.batch_size, len(self.replay))))
            )
        arena = Arena(
            simulations=self.config.simulations,
            board_size=self.config.board_size,
            walls_per_player=self.config.walls_per_player,
            max_plies=self.config.max_plies,
            rng=random.Random(self.rng.getrandbits(64)),
        )
        result: ArenaResult = arena.play_match(
            NeuralEvaluator(candidate, self.device),
            NeuralEvaluator(self.champion, self.device),
            self.config.arena_games,
        )
        promoted = result.candidate_score >= self.config.promotion_score
        if promoted:
            self.champion = candidate
            optimizer = learner.optimizer
        else:
            optimizer = Learner(self.champion, learning_rate=self.config.learning_rate, device=self.device).optimizer
        self.generation += 1
        summary = {
            "generation": self.generation,
            "new_examples": added,
            "replay_examples": len(self.replay),
            "game_lengths": game_lengths,
            "initial_loss": metrics[0]["loss"],
            "final_loss": metrics[-1]["loss"],
            "candidate_wins": result.candidate_wins,
            "champion_wins": result.champion_wins,
            "draws": result.draws,
            "candidate_score": result.candidate_score,
            "promoted": promoted,
        }
        path = self.checkpoint_dir / f"generation_{self.generation:03d}.pt"
        save_checkpoint(
            path,
            self.champion,
            optimizer,
            self.generation,
            metadata={"summary": summary, "config": asdict(self.config)},
        )
        summary["checkpoint"] = str(path)
        return summary