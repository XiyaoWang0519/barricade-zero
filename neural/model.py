"""Small residual policy-value network for configurable board sizes."""

from __future__ import annotations

import torch
from torch import nn

from barricade.actions import action_size


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.body(inputs))


class PolicyValueNetwork(nn.Module):
    def __init__(self, board_size: int = 9, channels: int = 64, residual_blocks: int = 4) -> None:
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size(board_size)
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.stem = nn.Sequential(
            nn.Conv2d(8, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*(ResidualBlock(channels) for _ in range(residual_blocks)))
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.GroupNorm(1, 2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, self.action_size),
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.GroupNorm(1, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(board_size * board_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.tower(self.stem(inputs))
        return self.policy_head(features), self.value_head(self.value_conv(features))


def masked_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    if not torch.all(legal_mask.any(dim=-1)):
        raise ValueError("every position must have at least one legal action")
    return torch.softmax(logits.masked_fill(~legal_mask, float("-inf")), dim=-1)
