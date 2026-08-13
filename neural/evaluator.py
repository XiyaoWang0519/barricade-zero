"""Adapter exposing a PyTorch policy-value network to reference MCTS."""

from __future__ import annotations

import torch

from barricade.encoding import legal_action_mask, rotate_policy
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

    @torch.inference_mode()
    def evaluate_batch(
        self, states: list[GameState], mask_legal: bool = True
    ) -> list[tuple[list[float], float]]:
        if not states:
            return []
        canonicals = [state.canonical() for state in states]
        inputs = torch.tensor(
            [self.encoding_backend.encode_state(state) for state in canonicals],
            dtype=torch.float32,
            device=self.device,
        )
        self.model.eval()
        logits, values = self.model(inputs)
        if mask_legal:
            masks = torch.tensor(
                [legal_action_mask(state) for state in canonicals],
                dtype=torch.bool,
                device=self.device,
            )
            policies = masked_softmax(logits, masks).cpu().tolist()
        else:
            policies = torch.softmax(logits, dim=-1).cpu().tolist()
        self.forward_calls += 1
        self.positions_evaluated += len(states)
        results = []
        for original, policy, value in zip(states, policies, values[:, 0].cpu().tolist()):
            if original.turn == 1:
                policy = rotate_policy(policy, original.size)
            results.append((policy, float(value)))
        return results

    def evaluate(self, state: GameState) -> tuple[list[float], float]:
        return self.evaluate_batch([state])[0]
