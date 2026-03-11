from __future__ import annotations

import os
from typing import Any

import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_policy(model: torch.nn.Module, path: str) -> None:
    """Save policy parameters to the given path."""
    directory = os.path.dirname(path)
    if directory:
        ensure_dir(directory)
    torch.save(model.state_dict(), path)


def load_policy(model: torch.nn.Module, path: str, map_location: Any | None = "cpu") -> None:
    """Load policy parameters from the given checkpoint path."""
    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)

