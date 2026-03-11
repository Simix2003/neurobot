from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from agents.brain import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE


class PolicyNetwork(nn.Module):
    """Stochastic policy over continuous actions for NeuroBot.

    Observations: 5D sensor vector.
    Actions: 2D continuous vector in [-1, 1]^2 (forward, turn).
    Also predicts a scalar state value for advantage-based updates.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.value_head = nn.Linear(HIDDEN_SIZE, 1)
        # Learnable log standard deviation per action dimension
        self.log_std = nn.Parameter(torch.zeros(OUTPUT_SIZE))

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs)

    def _dist(self, features: torch.Tensor) -> Normal:
        mean = torch.tanh(self.mean_head(features))  # keep means in [-1, 1]
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or compute an action, its log-prob, and the state value.

        Returns:
            action: tensor of shape (2,) in approximately [-1, 1].
            log_prob: scalar tensor log probability of the action.
            value: scalar tensor predicted state value V(s).
        """
        features = self._features(obs)
        dist = self._dist(features)

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()

        # Clamp action into [-1, 1] range expected by the environment
        action = torch.clamp(action, -1.0, 1.0)

        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.value_head(features).squeeze(-1)

        # Remove batch dimension if present
        if action.dim() == 2 and action.size(0) == 1:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            value = value.squeeze(0)

        return action, log_prob, value


