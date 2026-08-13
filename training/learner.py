"""Replay storage and policy-value optimization."""

from __future__ import annotations

import random
from collections import deque
from typing import Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as functional

from barricade.backend import load_rules_backend
from barricade.self_play import TrainingExample


class ReplayBuffer:
    def __init__(self, capacity: int, rng: random.Random | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._items = deque(maxlen=capacity)
        self.rng = rng or random.Random()

    def extend(self, examples: Iterable) -> None:
        self._items.extend(examples)

    def sample(self, batch_size: int) -> list:
        if not 0 < batch_size <= len(self._items):
            raise ValueError("invalid batch size")
        return self.rng.sample(list(self._items), batch_size)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator:
        return iter(self._items)


class Learner:
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        value_weight: float = 1.0,
        gradient_clip: float = 1.0,
        device: str | torch.device = "cpu",
        encoding_backend=None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.value_weight = value_weight
        self.gradient_clip = gradient_clip
        self.encoding_backend = encoding_backend or load_rules_backend()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def _batch(self, examples: list[TrainingExample]) -> tuple[torch.Tensor, ...]:
        states = torch.from_numpy(
            self.encoding_backend.encode_batch([example.state for example in examples])
        ).to(self.device)
        policies = torch.from_numpy(
            np.asarray([example.policy for example in examples], dtype=np.float32)
        ).to(self.device)
        outcomes = torch.from_numpy(
            np.asarray([[example.outcome] for example in examples], dtype=np.float32)
        ).to(self.device)
        return states, policies, outcomes

    def train_batch(self, examples: list[TrainingExample]) -> dict[str, float]:
        self.model.train()
        states, target_policies, target_values = self._batch(examples)
        logits, values = self.model(states)
        policy_loss = -(target_policies * functional.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        value_loss = functional.mse_loss(values, target_values)
        loss = policy_loss + self.value_weight * value_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()
        return {
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "gradient_norm": float(gradient_norm.detach()),
        }
