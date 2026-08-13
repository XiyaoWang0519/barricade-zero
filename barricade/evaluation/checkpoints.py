"""Checkpoint inspection and player construction for evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from neural.evaluator import NeuralEvaluator
from neural.model import PolicyValueNetwork

from .players import SearchPlayerFactory


@dataclass(frozen=True)
class NetworkSpec:
    board_size: int
    channels: int
    residual_blocks: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointInfo:
    path: str
    sha256: str
    model_sha256: str
    generation: int | None
    spec: NetworkSpec
    walls_per_player: int | None

    def to_dict(self) -> dict:
        result = asdict(self)
        result["spec"] = self.spec.to_dict()
        return result


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_dict(payload: dict) -> dict:
    if "model" in payload and hasattr(payload["model"], "keys"):
        return payload["model"]
    if "stem.0.weight" in payload:
        return payload
    raise ValueError("checkpoint does not contain a PolicyValueNetwork state dict")


def state_dict_sha256(state_dict: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def infer_network_spec(state_dict: dict) -> NetworkSpec:
    try:
        channels = int(state_dict["stem.0.weight"].shape[0])
        squares = int(state_dict["value_head.0.weight"].shape[1])
    except (KeyError, IndexError) as error:
        raise ValueError("checkpoint is not a PolicyValueNetwork state dict") from error
    board_size = int(squares**0.5)
    if board_size * board_size != squares:
        raise ValueError("could not infer board size from value head")
    block_indices = {
        int(parts[1])
        for key in state_dict
        if key.startswith("tower.")
        and len(parts := key.split(".")) > 1
        and parts[1].isdigit()
    }
    residual_blocks = max(block_indices) + 1 if block_indices else 0
    return NetworkSpec(board_size, channels, residual_blocks)


def load_checkpoint_evaluator(
    path: str | Path,
    *,
    device: str = "cpu",
    mixed_precision: bool = False,
) -> tuple[NeuralEvaluator, CheckpointInfo]:
    checkpoint_path = Path(path).resolve()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dict")
    state_dict = model_state_dict(payload)
    spec = infer_network_spec(state_dict)
    model = PolicyValueNetwork(
        spec.board_size,
        channels=spec.channels,
        residual_blocks=spec.residual_blocks,
    )
    model.load_state_dict(state_dict)
    model.eval()
    metadata = payload.get("metadata") or {}
    config = metadata.get("config") if isinstance(metadata, dict) else None
    walls = config.get("walls_per_player") if isinstance(config, dict) else None
    generation = payload.get("generation")
    info = CheckpointInfo(
        path=str(checkpoint_path),
        sha256=file_sha256(checkpoint_path),
        model_sha256=state_dict_sha256(state_dict),
        generation=int(generation) if generation is not None else None,
        spec=spec,
        walls_per_player=int(walls) if walls is not None else None,
    )
    return (
        NeuralEvaluator(
            model,
            device=device,
            mixed_precision=mixed_precision,
        ),
        info,
    )


def checkpoint_player_factory(
    path: str | Path,
    *,
    simulations: int,
    device: str = "cpu",
    mixed_precision: bool = False,
    label: str | None = None,
) -> tuple[SearchPlayerFactory, CheckpointInfo]:
    evaluator, info = load_checkpoint_evaluator(
        path,
        device=device,
        mixed_precision=mixed_precision,
    )
    name = label or Path(path).stem
    return SearchPlayerFactory(name, evaluator, simulations), info


def untrained_player_factory(
    spec: NetworkSpec,
    *,
    simulations: int,
    seed: int,
    device: str = "cpu",
    mixed_precision: bool = False,
) -> SearchPlayerFactory:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = PolicyValueNetwork(
            spec.board_size,
            channels=spec.channels,
            residual_blocks=spec.residual_blocks,
        )
    evaluator = NeuralEvaluator(
        model,
        device=device,
        mixed_precision=mixed_precision,
    )
    return SearchPlayerFactory(f"untrained_seed_{seed}", evaluator, simulations)
