"""Adapter exposing a PyTorch policy-value network to reference MCTS."""

from __future__ import annotations

import torch

from barricade.encoding import encode_state, legal_action_mask, rotate_policy
from barricade.state import GameState
from .model import masked_softmax


class NeuralEvaluator:
    def __init__(self, model: torch.nn.Module, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)

    @torch.inference_mode()
    def evaluate(self, state: GameState) -> tuple[list[float], float]:
        canonical = state.canonical()
        inputs = torch.tensor([encode_state(canonical)], dtype=torch.float32, device=self.device)
        logits, value = self.model(inputs)
        canonical_mask = torch.tensor(
            [legal_action_mask(canonical)], dtype=torch.bool, device=self.device
        )
        policy = masked_softmax(logits, canonical_mask)[0].cpu().tolist()
        if state.turn == 1:
            policy = rotate_policy(policy, state.size)
        return policy, float(value[0, 0].cpu())