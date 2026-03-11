from __future__ import annotations

import pygame

from config import BACKGROUND_COLOR
from world.environment import Environment


class Renderer:
    """Minimal renderer for Step 1A.

    Responsible for clearing the screen, asking the environment to draw itself,
    and flipping the display buffer.
    """

    def __init__(self, screen: pygame.Surface, environment: Environment) -> None:
        self._screen = screen
        self._environment = environment

    def render_frame(self) -> None:
        """Render a single frame of the world."""
        self._screen.fill(BACKGROUND_COLOR)
        self._environment.draw(self._screen)
        pygame.display.flip()

