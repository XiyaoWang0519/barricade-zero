"""Small, resumable AlphaZero generation loop."""

from __future__ import annotations

import copy
import hashlib
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from barricade.arena import Arena, ArenaResult
from barricade.concurrent_self_play import play_concurrent_games
from neural.evaluator import NeuralEvaluator
from neural.model import PolicyValueNetwork
from .checkpoint import load_checkpoint, save_checkpoint
from .learner import Learner, ReplayBuffer


def model_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    mixed_precision: bool = False
    torch_threads: int | None = None


class GenerationTrainer:
    def __init__(self, config: GenerationConfig, checkpoint_dir: str | Path, device: str = "cpu") -> None:
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.rng = random.Random(config.seed)
        if config.torch_threads is not None:
            if config.torch_threads <= 0:
                raise ValueError("torch_threads must be positive")
            torch.set_num_threads(config.torch_threads)
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
        training_state = payload.get("metadata", {}).get("training_state", {})
        replay = training_state.get("replay")
        if replay is not None:
            self.replay = ReplayBuffer(self.config.replay_capacity, self.rng)
            self.replay.extend(replay)
        rng_state = training_state.get("rng_state")
        if rng_state is not None:
            self.rng.setstate(rng_state)

    def _self_play(self) -> tuple[int, list[int]]:
        evaluator = NeuralEvaluator(
            self.champion,
            self.device,
            mixed_precision=self.config.mixed_precision,
        )
        result = play_concurrent_games(
            evaluator,
            games=self.config.self_play_games,
            simulations=self.config.simulations,
            board_size=self.config.board_size,
            walls_per_player=self.config.walls_per_player,
            rng=random.Random(self.rng.getrandbits(64)),
            max_plies=self.config.max_plies,
        )
        self.replay.extend(result.examples)
        self._last_inference_metrics = {
            "inference_forward_calls": result.forward_calls,
            "positions_evaluated": result.positions_evaluated,
            "average_inference_batch_size": result.average_inference_batch_size,
            "self_play_draws": result.draws,
            "reused_root_visits": result.reused_root_visits,
        }
        return len(result.examples), result.game_lengths

    def run_generation(self) -> dict:
        generation_started = time.perf_counter()
        phase_started = generation_started
        added, game_lengths = self._self_play()
        self_play_seconds = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        candidate = copy.deepcopy(self.champion)
        initial_candidate_sha256 = model_sha256(candidate)
        learner = Learner(candidate, learning_rate=self.config.learning_rate, device=self.device)
        candidate_setup_seconds = time.perf_counter() - phase_started
        metrics = []
        phase_started = time.perf_counter()
        for _ in range(self.config.training_steps):
            metrics.append(
                learner.train_batch(self.replay.sample(min(self.config.batch_size, len(self.replay))))
            )
        training_seconds = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        trained_candidate_sha256 = model_sha256(candidate)
        candidate_hash_seconds = time.perf_counter() - phase_started
        arena = Arena(
            simulations=self.config.simulations,
            board_size=self.config.board_size,
            walls_per_player=self.config.walls_per_player,
            max_plies=self.config.max_plies,
            rng=random.Random(self.rng.getrandbits(64)),
        )
        phase_started = time.perf_counter()
        result: ArenaResult = arena.play_match(
            NeuralEvaluator(
                candidate,
                self.device,
                mixed_precision=self.config.mixed_precision,
            ),
            NeuralEvaluator(
                self.champion,
                self.device,
                mixed_precision=self.config.mixed_precision,
            ),
            self.config.arena_games,
        )
        arena_seconds = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        promoted = result.candidate_score >= self.config.promotion_score
        if promoted:
            self.champion = candidate
            optimizer = learner.optimizer
        else:
            optimizer = Learner(self.champion, learning_rate=self.config.learning_rate, device=self.device).optimizer
        promotion_seconds = time.perf_counter() - phase_started
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
            "initial_candidate_sha256": initial_candidate_sha256,
            "trained_candidate_sha256": trained_candidate_sha256,
            **self._last_inference_metrics,
            "self_play_seconds": self_play_seconds,
            "candidate_setup_seconds": candidate_setup_seconds,
            "training_seconds": training_seconds,
            "candidate_hash_seconds": candidate_hash_seconds,
            "arena_seconds": arena_seconds,
            "promotion_seconds": promotion_seconds,
        }
        path = self.checkpoint_dir / f"generation_{self.generation:03d}.pt"
        phase_started = time.perf_counter()
        save_checkpoint(
            path,
            self.champion,
            optimizer,
            self.generation,
            metadata={
                "summary": summary,
                "config": asdict(self.config),
                "training_state": {
                    "replay": list(self.replay),
                    "rng_state": self.rng.getstate(),
                },
            },
        )
        summary["checkpoint"] = str(path)
        summary["checkpoint_sha256"] = file_sha256(path)
        checkpoint_seconds = time.perf_counter() - phase_started
        generation_seconds = time.perf_counter() - generation_started
        summary["checkpoint_seconds"] = checkpoint_seconds
        summary["generation_seconds"] = generation_seconds
        summary["generation_overhead_seconds"] = generation_seconds - (
            self_play_seconds
            + candidate_setup_seconds
            + training_seconds
            + candidate_hash_seconds
            + arena_seconds
            + promotion_seconds
            + checkpoint_seconds
        )
        return summary
