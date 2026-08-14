"""Discover and load policy-value checkpoints for the gameplay UI."""

from __future__ import annotations

from pathlib import Path

import torch

from neural.evaluator import NeuralEvaluator
from neural.model import PolicyValueNetwork


def infer_network_spec(state_dict: dict) -> dict:
    if "stem.0.weight" not in state_dict or "value_head.0.weight" not in state_dict:
        raise ValueError("checkpoint is not a PolicyValueNetwork state dict")
    channels = int(state_dict["stem.0.weight"].shape[0])
    tower_indices = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("tower.") and key.split(".")[1].isdigit()
    }
    residual_blocks = max(tower_indices) + 1 if tower_indices else 0
    squares = int(state_dict["value_head.0.weight"].shape[1])
    board_size = int(squares**0.5)
    if board_size * board_size != squares:
        raise ValueError("could not infer board size from value head")
    return {
        "board_size": board_size,
        "channels": channels,
        "residual_blocks": residual_blocks,
    }


def _model_state_dict(payload: dict) -> dict:
    if "model" in payload and hasattr(payload["model"], "keys"):
        return payload["model"]
    if "stem.0.weight" in payload:
        return payload
    raise ValueError("checkpoint does not contain a model state dict")


def inspect_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dict")
    spec = infer_network_spec(_model_state_dict(payload))
    metadata = payload.get("metadata") or {}
    config = metadata.get("config") if isinstance(metadata, dict) else None
    summary = metadata.get("summary") if isinstance(metadata, dict) else None
    walls = None
    if isinstance(config, dict):
        walls = config.get("walls_per_player")
    return {
        "path": str(path),
        "name": path.name,
        "ok": True,
        "generation": payload.get("generation"),
        "board_size": spec["board_size"],
        "channels": spec["channels"],
        "residual_blocks": spec["residual_blocks"],
        "walls_per_player": walls,
        "summary": summary,
        "config": config,
    }


def list_checkpoints(directory: str | Path) -> list[dict]:
    root = Path(directory)
    if not root.exists():
        return []
    results = []
    for path in sorted(root.rglob("*.pt")):
        try:
            info = inspect_checkpoint(path)
            info["path"] = str(path.relative_to(root))
            results.append(info)
        except Exception as error:
            results.append(
                {
                    "path": str(path.relative_to(root)),
                    "name": path.name,
                    "ok": False,
                    "error": str(error),
                }
            )
    return results


def load_evaluator(path: str | Path, device: str = "cpu") -> tuple[NeuralEvaluator, dict]:
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    state_dict = _model_state_dict(payload)
    spec = infer_network_spec(state_dict)
    model = PolicyValueNetwork(
        spec["board_size"], spec["channels"], spec["residual_blocks"]
    )
    model.load_state_dict(state_dict)
    model.eval()
    info = inspect_checkpoint(path, map_location=device)
    return NeuralEvaluator(model, device), info
