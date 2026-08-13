"""Adapter exposing a PyTorch policy-value network to reference MCTS."""

from __future__ import annotations

from contextlib import nullcontext
import time

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
        mixed_precision: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.mixed_precision = bool(mixed_precision) and self.device.type == "cuda"
        self.forward_calls = 0
        self.positions_evaluated = 0
        self.batch_sizes: list[int] = []
        self.model_forward_seconds = 0.0
        self.host_to_device_transfers = 0
        self.host_to_device_seconds = 0.0
        self.device_synchronization_calls = 0
        self.device_synchronization_seconds = 0.0
        self.encoding_backend = encoding_backend or load_rules_backend()

    @property
    def average_batch_size(self) -> float:
        return self.positions_evaluated / self.forward_calls if self.forward_calls else 0.0

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            started = time.perf_counter()
            torch.cuda.synchronize(self.device)
            self.device_synchronization_calls += 1
            self.device_synchronization_seconds += time.perf_counter() - started

    def _move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device == self.device:
            return tensor
        started = time.perf_counter()
        self.host_to_device_transfers += 1
        result = tensor.to(
            self.device,
            non_blocking=self.device.type == "cuda",
        )
        self.host_to_device_seconds += time.perf_counter() - started
        return result

    def encode_inputs(self, states: list[GameState], canonical: bool = False):
        encoder = (
            self.encoding_backend.encode_canonical_batch
            if canonical and hasattr(self.encoding_backend, "encode_canonical_batch")
            else self.encoding_backend.encode_batch
        )
        array = encoder(states)
        tensor = torch.from_numpy(array)
        return array, self._move_to_device(tensor)

    @torch.inference_mode()
    def _evaluate_input_arrays(
        self, states: list[GameState] | None, inputs: torch.Tensor,
        canonicals: list[GameState] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.model.training:
            self.model.eval()
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.mixed_precision
            else nullcontext()
        )
        self._synchronize()
        started = time.perf_counter()
        with autocast:
            logits, values = self.model(inputs)
        self._synchronize()
        self.model_forward_seconds += time.perf_counter() - started
        logits = logits.float()
        values = values.float()
        if canonicals is not None:
            mask_array = np.asarray(
                [legal_action_mask(state) for state in canonicals], dtype=np.bool_
            )
            masks = self._move_to_device(torch.from_numpy(mask_array))
            policies = masked_softmax(logits, masks)
        else:
            policies = torch.softmax(logits, dim=-1)
        self.forward_calls += 1
        batch_size = int(inputs.shape[0])
        self.positions_evaluated += batch_size
        self.batch_sizes.append(batch_size)
        policy_array = policies.cpu().numpy()
        if states is not None:
            for index, original in enumerate(states):
                if original.turn == 1:
                    indices = np.asarray(
                        policy_rotation_indices(original.size), dtype=np.intp
                    )
                    policy_array[index] = policy_array[index][indices]
        return (
            np.ascontiguousarray(policy_array, dtype=np.float32),
            np.ascontiguousarray(values[:, 0].cpu().numpy(), dtype=np.float32),
        )

    @torch.inference_mode()
    def evaluate_batch_arrays(
        self, states: list[GameState], mask_legal: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        if not states:
            return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.float32)
        canonicals = [state.canonical() for state in states] if mask_legal else None
        _array, inputs = self.encode_inputs(
            states if canonicals is None else canonicals,
            canonical=canonicals is None,
        )
        return self._evaluate_input_arrays(states, inputs, canonicals)

    @torch.inference_mode()
    def evaluate_prepared_batch_arrays(
        self, states: list[GameState]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not states:
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.zeros(1, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )
        encoded, offsets, actions = self.encoding_backend.prepare_batch(states)
        inputs = self._move_to_device(torch.from_numpy(encoded))
        policies, values = self._evaluate_input_arrays(states, inputs)
        return policies, values, offsets, actions

    @torch.inference_mode()
    def evaluate_encoded_batch_arrays(
        self, encoded: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if encoded.shape[0] == 0:
            return (
                np.empty((0, self.model.action_size), dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )
        inputs = self._move_to_device(torch.from_numpy(encoded))
        return self._evaluate_input_arrays(None, inputs)

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
