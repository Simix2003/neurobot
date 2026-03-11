from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


INPUT_SIZE = 5  # [food_distance, food_angle, wall_front, wall_left, wall_right]
HIDDEN_SIZE = 32
OUTPUT_SIZE = 2  # [forward_value, turn_value]


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
            nn.Tanh(),  # keep outputs in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


@dataclass
class Brain:
    """Small neural controller for the robot.

    For Step 2, weights are random and never trained; the goal is to
    prove the interface from sensors -> brain -> actions.
    """

    model: _MLP = _MLP()

    def forward(self, inputs: np.ndarray) -> Tuple[float, float]:
        """Compute movement outputs from sensor inputs.

        Args:
            inputs: shape (5,) array of normalized sensor values.
        Returns:
            (forward_value, turn_value) in roughly [-1, 1].
        """
        if inputs.shape != (INPUT_SIZE,):
            raise ValueError(f"Expected inputs shape ({INPUT_SIZE},), got {inputs.shape}")

        x = torch.from_numpy(inputs.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            y = self.model(x)
        forward, turn = y.squeeze(0).tolist()
        return float(forward), float(turn)

