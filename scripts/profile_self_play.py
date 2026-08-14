#!/usr/bin/env python3
"""Reproducible cProfile harness for rules, MCTS, and batched inference."""

import argparse
import cProfile
import io
import json
import platform
import pstats
import random
import resource
import statistics
import sys
import threading
import time
from pathlib import Path

import torch

from barricade.concurrent_self_play import play_concurrent_games
from barricade.backend import load_rules_backend
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


def _distribution(values: list[float | int]) -> dict:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "maximum": None,
            "population_stddev": None,
            "median_absolute_deviation": None,
            "first_quartile": None,
            "third_quartile": None,
            "interquartile_range": None,
        }
    median = statistics.median(values)
    if len(values) == 1:
        first_quartile = third_quartile = median
    else:
        first_quartile, _, third_quartile = statistics.quantiles(
            values, n=4, method="inclusive"
        )
    return {
        "count": len(values),
        "minimum": min(values),
        "median": median,
        "maximum": max(values),
        "population_stddev": statistics.pstdev(values),
        "median_absolute_deviation": statistics.median(
            [abs(value - median) for value in values]
        ),
        "first_quartile": first_quartile,
        "third_quartile": third_quartile,
        "interquartile_range": third_quartile - first_quartile,
    }


def summarize_runs(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("at least one measured run is required")
    metrics = (
        "seconds",
        "setup_seconds",
        "model_forward_seconds",
        "non_model_seconds",
        "positions_per_second",
        "examples_per_second",
        "positions",
        "examples",
        "forward_calls",
        "average_batch_size",
        "draws",
        "reused_root_visits",
    )
    summary = {"repetitions": len(runs)}
    for metric in metrics:
        values = [run[metric] for run in runs if metric in run]
        if values:
            summary[metric] = _distribution(values)
    return summary


def run_workload(
    board_size: int, walls: int, games: int, simulations: int, seed: int,
    channels: int = 16, residual_blocks: int = 2,
    device: str | torch.device = "cpu", mixed_precision: bool = False,
    collect_metrics: bool = False,
    use_native_search: bool | None = None,
    torch_threads: int | None = None,
    cuda_graphs: bool = False,
) -> dict:
    if torch_threads is not None:
        if torch_threads <= 0:
            raise ValueError("torch_threads must be positive")
        torch.set_num_threads(torch_threads)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    setup_started = time.perf_counter()
    torch.manual_seed(seed)
    backend = load_rules_backend(collect_metrics=collect_metrics)
    evaluator = NeuralEvaluator(
        PolicyValueNetwork(board_size, channels=channels, residual_blocks=residual_blocks),
        device=device,
        encoding_backend=backend,
        mixed_precision=mixed_precision,
        cuda_graphs=cuda_graphs,
        cuda_graph_max_batch_size=games if cuda_graphs else None,
    )
    setup_seconds = time.perf_counter() - setup_started
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
            rules_backend=backend,
            use_native_search=use_native_search,
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
    summary = {
        "board_size": board_size,
        "walls": walls,
        "games": games,
        "simulations": simulations,
        "channels": channels,
        "residual_blocks": residual_blocks,
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "precision": "float16" if evaluator.mixed_precision else "float32",
        "cuda_graphs": evaluator.cuda_graphs,
        "gpu_model": gpu_model,
        "gpu_vram_total_bytes": gpu_vram,
        "examples": len(result.examples),
        "examples_per_second": len(result.examples) / elapsed,
        "draws": result.draws,
        "wins": list(result.wins),
        "seconds": elapsed,
        "setup_seconds": setup_seconds,
        "positions": result.positions_evaluated,
        "forward_calls": result.forward_calls,
        "average_batch_size": result.average_inference_batch_size,
        "positions_per_second": result.positions_evaluated / elapsed,
        "model_forward_seconds": evaluator.model_forward_seconds,
        "non_model_seconds": elapsed - evaluator.model_forward_seconds,
        "peak_gpu_allocated_bytes": peak_allocated,
        "peak_gpu_reserved_bytes": peak_reserved,
        "gpu_utilization_samples": len(sampler.samples),
        "gpu_utilization_average_percent": (
            sum(sampler.samples) / len(sampler.samples) if sampler.samples else None
        ),
        "gpu_utilization_max_percent": max(sampler.samples) if sampler.samples else None,
        "reused_root_visits": result.reused_root_visits,
        "search_backend": result.search_backend,
        "native_search_boundary_calls": result.native_search_boundary_calls,
        "native_search_boundary_seconds": result.native_search_boundary_seconds,
        "native_preparation_seconds": result.native_preparation_seconds,
        "native_encoding_seconds": result.native_encoding_seconds,
        "native_legality_seconds": result.native_legality_seconds,
        "rules_backend": backend.name,
        "python_native_boundary_calls": (
            backend.boundary_calls + result.native_search_boundary_calls
        ),
        "python_native_boundary_seconds": (
            backend.boundary_seconds + result.native_search_boundary_seconds
        ),
        "legal_action_calls": backend.legal_action_calls,
        "legal_action_seconds": backend.legal_action_seconds,
        "path_validation_calls": backend.path_validation_calls,
        "path_validation_seconds": backend.path_validation_seconds,
        "encoding_calls": backend.encoding_calls,
        "encoding_seconds": backend.encoding_seconds,
        "batch_preparation_calls": backend.batch_preparation_calls,
        "batch_preparation_seconds": backend.batch_preparation_seconds,
        "tree_nodes_created": result.tree_nodes_created,
        "tree_expansions": result.tree_expansions,
        "tree_traversal_seconds": result.tree_traversal_seconds,
        "expansion_seconds": result.expansion_seconds,
        "host_to_device_transfers": evaluator.host_to_device_transfers,
        "host_to_device_seconds": evaluator.host_to_device_seconds,
        "device_synchronization_calls": evaluator.device_synchronization_calls,
        "device_synchronization_seconds": evaluator.device_synchronization_seconds,
        "peak_process_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1024 if sys.platform.startswith("linux") else 1)
        ),
        "game_length_count": len(result.game_lengths),
        "game_length_min": min(result.game_lengths),
        "game_length_median": statistics.median(result.game_lengths),
        "game_length_max": max(result.game_lengths),
        "inference_batch_size_count": len(evaluator.batch_sizes),
        "inference_batch_size_min": min(evaluator.batch_sizes),
        "inference_batch_size_median": statistics.median(evaluator.batch_sizes),
        "inference_batch_size_max": max(evaluator.batch_sizes),
    }
    return summary


def _environment_summary() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
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
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--python-search", action="store_true")
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--profile")
    parser.add_argument("--json-output")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")

    workload = (
        args.board_size, args.walls, args.games, args.simulations, args.seed,
        args.channels, args.blocks, args.device, args.mixed_precision,
    )
    workload_keywords = {
        "use_native_search": False if args.python_search else None,
        "torch_threads": args.torch_threads,
        "cuda_graphs": args.cuda_graphs,
    }
    for index in range(args.warmups):
        run_workload(*workload, **workload_keywords)
        print(f"completed warmup {index + 1}/{args.warmups}", file=sys.stderr, flush=True)

    runs = []
    for index in range(args.repetitions):
        runs.append(run_workload(*workload, **workload_keywords))
        print(
            f"completed measured repetition {index + 1}/{args.repetitions}",
            file=sys.stderr,
            flush=True,
        )

    report = {
        "environment": _environment_summary(),
        "warmups": args.warmups,
        "runs": runs,
        "aggregate": summarize_runs(runs),
    }
    profile_text = None
    if args.profile:
        profile_path = Path(args.profile)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profiler = cProfile.Profile()
        profiler.enable()
        profile_summary = run_workload(
            *workload, collect_metrics=True, **workload_keywords
        )
        profiler.disable()
        profiler.dump_stats(profile_path)
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative")
        stats.print_stats(args.top)
        profile_summary["profile"] = str(profile_path)
        report["profile_run"] = profile_summary
        profile_text = stream.getvalue()
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if profile_text is not None:
        print(profile_text)


if __name__ == "__main__":
    main()
