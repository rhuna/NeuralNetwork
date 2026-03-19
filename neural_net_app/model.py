from __future__ import annotations

import torch
from torch import nn


class DigitClassifier(nn.Module):
    def __init__(self, input_size: int = 64, num_classes: int = 10) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)