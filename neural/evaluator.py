"""Adapter exposing a PyTorch policy-value network to reference MCTS."""

from __future__ import annotations

import torch
import numpy as np

from barricade.encoding import legal_action_mask, policy_rotation_indices
from barricade.backend import load_rules_backend
from barricade.state import GameState
from .model import masked_softmax


class NeuralEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device = "cpu",
        encoding_backend=None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.forward_calls = 0
        self.positions_evaluated = 0
        self.encoding_backend = encoding_backend or load_rules_backend()

    @property
    def average_batch_size(self) -> float:
        return self.positions_evaluated / self.forward_calls if self.forward_calls else 0.0

    def encode_inputs(self, states: list[GameState]):
        array = self.encoding_backend.encode_batch(states)
        tensor = torch.from_numpy(array)
        if self.device.type != "cpu":
            tensor = tensor.to(self.device)
        return array, tensor

    @torch.inference_mode()
    def evaluate_batch_arrays(
        self, states: list[GameState], mask_legal: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        if not states:
            return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.float32)
        canonicals = [state.canonical() for state in states]
        _array, inputs = self.encode_inputs(canonicals)
        self.model.eval()
        logits, values = self.model(inputs)
        if mask_legal:
            masks = torch.tensor(
                [legal_action_mask(state) for state in canonicals],
                dtype=torch.bool,
                device=self.device,
            )
            policies = masked_softmax(logits, masks)
        else:
            policies = torch.softmax(logits, dim=-1)
        self.forward_calls += 1
        self.positions_evaluated += len(states)
        policy_array = policies.cpu().numpy()
        for index, original in enumerate(states):
            if original.turn == 1:
                indices = np.asarray(
                    policy_rotation_indices(original.size), dtype=np.intp
                )
                policy_array[index] = policy_array[index][indices]
        return np.ascontiguousarray(policy_array), values[:, 0].cpu().numpy()

    @torch.inference_mode()
    def evaluate_batch(
        self, states: list[GameState], mask_legal: bool = True
    ) -> list[tuple[list[float], float]]:
        policies, values = self.evaluate_batch_arrays(states, mask_legal)
        return [
            (policy.tolist(), float(value))
            for policy, value in zip(policies, values)
        ]

    def evaluate(self, state: GameState) -> tuple[list[float], float]:
        return self.evaluate_batch([state])[0]
