from __future__ import annotations

import pygame

from config import BACKGROUND_COLOR, SCORE_TEXT_COLOR
from world.environment import Environment


class Renderer:
    """Minimal renderer for Step 1A.

    Responsible for clearing the screen, asking the environment to draw itself,
    and flipping the display buffer.
    """

    def __init__(self, screen: pygame.Surface, environment: Environment) -> None:
        self._screen = screen
        self._environment = environment
        self._font: pygame.font.Font | None = None

    def render_frame(self) -> None:
        """Render a single frame of the world."""
        self._screen.fill(BACKGROUND_COLOR)
        self._environment.draw(self._screen)
        self._draw_hud()
        pygame.display.flip()

    def _draw_hud(self) -> None:
        if self._font is None:
            self._font = pygame.font.SysFont("consolas", 18)

        score_text = f"Score: {self._environment.score}"
        surf = self._font.render(score_text, True, SCORE_TEXT_COLOR)
        self._screen.blit(surf, (10, 10))

