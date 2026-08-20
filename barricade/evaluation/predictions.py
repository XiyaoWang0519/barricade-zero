"""Held-out policy/value metrics independent of the training loop."""

from __future__ import annotations

import json
import inspect
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from barricade.backend import load_rules_backend
from barricade.state import GameState


@dataclass(frozen=True)
class LabeledPosition:
    position_id: str
    state: GameState
    target_policy: Sequence[float]
    target_value: float

    def __post_init__(self) -> None:
        policy = [float(value) for value in self.target_policy]
        if len(policy) != self.state.action_size:
            raise ValueError("target policy has the wrong action size")
        if any(value < 0.0 or not math.isfinite(value) for value in policy):
            raise ValueError("target policy must contain finite non-negative values")
        if not math.isclose(sum(policy), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("target policy must sum to one")
        if not -1.0 <= float(self.target_value) <= 1.0:
            raise ValueError("target value must be between -1 and one")

    def to_dict(self) -> dict:
        return {
            "id": self.position_id,
            "state": self.state.to_dict(),
            "policy": list(self.target_policy),
            "value": float(self.target_value),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LabeledPosition":
        return cls(
            position_id=str(data["id"]),
            state=GameState.from_dict(data["state"]),
            target_policy=[float(value) for value in data["policy"]],
            target_value=float(data["value"]),
        )


@dataclass(frozen=True)
class PredictionMetrics:
    positions: int
    policy_cross_entropy: float
    policy_top1_accuracy: float
    policy_top3_accuracy: float
    value_mse: float
    value_mae: float
    value_calibration_error: float
    illegal_policy_mass: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_labeled_positions(path: str | Path) -> list[LabeledPosition]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["positions"] if isinstance(data, dict) else data
    positions = [LabeledPosition.from_dict(row) for row in rows]
    if not positions:
        raise ValueError("labeled position dataset is empty")
    return positions


def evaluate_predictions(
    evaluator,
    positions: Sequence[LabeledPosition],
    *,
    batch_size: int = 256,
) -> PredictionMetrics:
    if not positions:
        raise ValueError("at least one labeled position is required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    evaluate_batch = getattr(evaluator, "evaluate_batch", None)
    evaluate_arrays = getattr(evaluator, "evaluate_batch_arrays", None)
    supports_mask = False
    if evaluate_arrays is not None:
        try:
            supports_mask = "mask_legal" in inspect.signature(evaluate_arrays).parameters
        except (TypeError, ValueError):
            supports_mask = False
    elif evaluate_batch is not None:
        try:
            supports_mask = "mask_legal" in inspect.signature(evaluate_batch).parameters
        except (TypeError, ValueError):
            supports_mask = False
    else:
        raise TypeError("evaluator must provide evaluate_batch or evaluate_batch_arrays")
    backend = load_rules_backend()
    epsilon = 1e-12
    policy_rows: list[np.ndarray] = []
    value_rows: list[np.ndarray] = []
    target_policy_rows: list[np.ndarray] = []
    target_values = np.empty(len(positions), dtype=np.float64)
    illegal_mass = 0.0
    offset = 0
    for start in range(0, len(positions), batch_size):
        batch = positions[start : start + batch_size]
        states = [item.state for item in batch]
        if evaluate_arrays is not None:
            if supports_mask:
                policies, values = evaluate_arrays(states, mask_legal=False)
            else:
                policies, values = evaluate_arrays(states)
            policy_batch = np.asarray(policies, dtype=np.float64)
            value_batch = np.asarray(values, dtype=np.float64).reshape(-1)
        else:
            if supports_mask:
                evaluations = evaluate_batch(states, mask_legal=False)
            else:
                evaluations = evaluate_batch(states)
            policy_batch = np.asarray(
                [evaluation[0] for evaluation in evaluations], dtype=np.float64
            )
            value_batch = np.asarray(
                [evaluation[1] for evaluation in evaluations], dtype=np.float64
            )
        if policy_batch.shape[0] != len(batch):
            raise ValueError("evaluator returned the wrong number of policies")
        if policy_batch.shape[1] != batch[0].state.action_size:
            raise ValueError("evaluator returned a policy with the wrong action size")
        target_batch = np.asarray(
            [item.target_policy for item in batch], dtype=np.float64
        )
        offsets, actions = backend.legal_actions_batch(states)
        legal_mask = np.zeros(policy_batch.shape, dtype=bool)
        for index in range(len(batch)):
            legal_mask[index, actions[offsets[index] : offsets[index + 1]]] = True
        illegal_mass += float(policy_batch[~legal_mask].sum())
        policy_rows.append(policy_batch)
        value_rows.append(value_batch)
        target_policy_rows.append(target_batch)
        end = offset + len(batch)
        target_values[offset:end] = [item.target_value for item in batch]
        offset = end
    policies = np.concatenate(policy_rows, axis=0)
    values = np.concatenate(value_rows, axis=0)
    targets = np.concatenate(target_policy_rows, axis=0)
    safe_policies = np.maximum(policies, epsilon)
    cross_entropy = float(-(targets * np.log(safe_policies)).sum() / len(positions))
    predicted_best = policies.argmax(axis=1)
    target_best = targets.argmax(axis=1)
    top1_correct = int((predicted_best == target_best).sum())
    topk = min(3, policies.shape[1])
    top3 = np.argpartition(-policies, topk - 1, axis=1)[:, :topk]
    top3_correct = int((top3 == target_best[:, None]).any(axis=1).sum())
    errors = values - target_values
    count = len(positions)
    calibration_error = 0.0
    for bin_index in range(10):
        lower = -1.0 + 0.2 * bin_index
        upper = lower + 0.2
        selected = (values >= lower) & (values < upper)
        if bin_index == 9:
            selected |= values == 1.0
        if not selected.any():
            continue
        predicted_mean = float(values[selected].mean())
        target_mean = float(target_values[selected].mean())
        calibration_error += float(selected.mean()) * abs(predicted_mean - target_mean)
    return PredictionMetrics(
        positions=count,
        policy_cross_entropy=cross_entropy,
        policy_top1_accuracy=top1_correct / count,
        policy_top3_accuracy=top3_correct / count,
        value_mse=float(np.mean(errors * errors)),
        value_mae=float(np.mean(np.abs(errors))),
        value_calibration_error=float(calibration_error),
        illegal_policy_mass=illegal_mass / count,
    )
