"""Held-out policy/value metrics independent of the training loop."""

from __future__ import annotations

import json
import inspect
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

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
    cross_entropy = value_squared_error = value_absolute_error = 0.0
    top1_correct = top3_correct = 0
    illegal_mass = 0.0
    calibration_values = []
    epsilon = 1e-12
    for start in range(0, len(positions), batch_size):
        batch = positions[start : start + batch_size]
        evaluate_parameters = inspect.signature(evaluator.evaluate_batch).parameters
        if "mask_legal" in evaluate_parameters:
            evaluations = evaluator.evaluate_batch(
                [item.state for item in batch], mask_legal=False
            )
        else:
            evaluations = evaluator.evaluate_batch([item.state for item in batch])
        for item, (predicted_policy, predicted_value) in zip(batch, evaluations):
            policy = [float(value) for value in predicted_policy]
            if len(policy) != item.state.action_size:
                raise ValueError("evaluator returned a policy with the wrong action size")
            target = [float(value) for value in item.target_policy]
            cross_entropy += -sum(
                probability * math.log(max(policy[action], epsilon))
                for action, probability in enumerate(target)
                if probability > 0.0
            )
            target_best = max(range(len(target)), key=target.__getitem__)
            if max(range(len(policy)), key=policy.__getitem__) == target_best:
                top1_correct += 1
            if target_best in sorted(
                range(len(policy)), key=policy.__getitem__, reverse=True
            )[:3]:
                top3_correct += 1
            legal = set(item.state.legal_actions())
            illegal_mass += sum(
                probability
                for action, probability in enumerate(policy)
                if action not in legal
            )
            error = float(predicted_value) - float(item.target_value)
            value_squared_error += error * error
            value_absolute_error += abs(error)
            calibration_values.append(
                (float(predicted_value), float(item.target_value))
            )
    count = len(positions)
    calibration_error = 0.0
    for bin_index in range(10):
        lower = -1.0 + 0.2 * bin_index
        upper = lower + 0.2
        values = [
            (predicted, target)
            for predicted, target in calibration_values
            if lower <= predicted < upper
            or (bin_index == 9 and predicted == 1.0)
        ]
        if values:
            predicted_mean = sum(value[0] for value in values) / len(values)
            target_mean = sum(value[1] for value in values) / len(values)
            calibration_error += (
                len(values) / count * abs(predicted_mean - target_mean)
            )
    return PredictionMetrics(
        positions=count,
        policy_cross_entropy=cross_entropy / count,
        policy_top1_accuracy=top1_correct / count,
        policy_top3_accuracy=top3_correct / count,
        value_mse=value_squared_error / count,
        value_mae=value_absolute_error / count,
        value_calibration_error=calibration_error,
        illegal_policy_mass=illegal_mass / count,
    )
