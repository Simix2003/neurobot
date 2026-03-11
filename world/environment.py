from __future__ import annotations

from dataclasses import dataclass

import pygame

from config import WINDOW_WIDTH, WINDOW_HEIGHT, WORLD_BORDER_COLOR, WORLD_BORDER_THICKNESS


@dataclass
class Environment:
    """Minimal world container for Step 1A.

    For now it only represents the rectangular world area and exposes
    update/draw hooks that will host entities in later steps.
    """

    width: int = WINDOW_WIDTH
    height: int = WINDOW_HEIGHT

    def update(self, dt: float) -> None:
        """Advance world simulation by dt seconds.

        Step 1A keeps this as a placeholder so the loop structure is ready.
        """

    def draw(self, surface: pygame.Surface) -> None:
        """Draw world-level visuals (currently just the border)."""
        rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(
            surface,
            WORLD_BORDER_COLOR,
            rect,
            WORLD_BORDER_THICKNESS,
        )

