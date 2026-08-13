#!/usr/bin/env python3
"""Reproducible cProfile harness for rules, MCTS, and batched inference."""

import argparse
import cProfile
import io
import json
import pstats
import random
import threading
import time
from pathlib import Path

import torch

from barricade.concurrent_self_play import play_concurrent_games
from barricade.state import GameState
from neural.evaluator import NeuralEvaluator
from neural.model import PolicyValueNetwork


class CudaUtilizationSampler:
    def __init__(self, device: torch.device, interval_seconds: float = 0.25) -> None:
        self.device = device
        self.interval_seconds = interval_seconds
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.device.type != "cuda" or not hasattr(torch.cuda, "utilization"):
            return
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()

    def _sample_until_stopped(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(int(torch.cuda.utilization(self.device)))
            except (ImportError, RuntimeError, OSError):
                return
            self._stop.wait(self.interval_seconds)


def run_workload(
    board_size: int, walls: int, games: int, simulations: int, seed: int,
    channels: int = 16, residual_blocks: int = 2,
    device: str | torch.device = "cpu", mixed_precision: bool = False,
) -> dict:
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    torch.manual_seed(seed)
    evaluator = NeuralEvaluator(
        PolicyValueNetwork(board_size, channels=channels, residual_blocks=residual_blocks),
        device=device,
        mixed_precision=mixed_precision,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    sampler = CudaUtilizationSampler(device)
    sampler.start()
    started = time.perf_counter()
    try:
        result = play_concurrent_games(
            evaluator,
            games=games,
            simulations=simulations,
            board_size=board_size,
            walls_per_player=walls,
            rng=random.Random(seed),
            max_plies=300,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        sampler.stop()
    elapsed = time.perf_counter() - started
    gpu_model = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    gpu_vram = (
        torch.cuda.get_device_properties(device).total_memory
        if device.type == "cuda"
        else 0
    )
    peak_allocated = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    )
    return {
        "board_size": board_size,
        "walls": walls,
        "games": games,
        "simulations": simulations,
        "channels": channels,
        "residual_blocks": residual_blocks,
        "device": str(device),
        "precision": "float16" if evaluator.mixed_precision else "float32",
        "gpu_model": gpu_model,
        "gpu_vram_total_bytes": gpu_vram,
        "examples": len(result.examples),
        "draws": result.draws,
        "seconds": elapsed,
        "positions": result.positions_evaluated,
        "forward_calls": result.forward_calls,
        "average_batch_size": result.average_inference_batch_size,
        "positions_per_second": result.positions_evaluated / elapsed,
        "model_forward_seconds": evaluator.model_forward_seconds,
        "peak_gpu_allocated_bytes": peak_allocated,
        "peak_gpu_reserved_bytes": peak_reserved,
        "gpu_utilization_samples": len(sampler.samples),
        "gpu_utilization_average_percent": (
            sum(sampler.samples) / len(sampler.samples) if sampler.samples else None
        ),
        "gpu_utilization_max_percent": max(sampler.samples) if sampler.samples else None,
        "reused_root_visits": result.reused_root_visits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--walls", type=int, default=10)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--profile", default="profiles/self_play.prof")
    parser.add_argument("--json-output")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    profile_path = Path(args.profile)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    profiler.enable()
    summary = run_workload(
        args.board_size, args.walls, args.games, args.simulations, args.seed,
        args.channels, args.blocks, args.device, args.mixed_precision,
    )
    profiler.disable()
    profiler.dump_stats(profile_path)
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative")
    stats.print_stats(args.top)
    summary["profile"] = str(profile_path)
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    print(stream.getvalue())


if __name__ == "__main__":
    main()
