from __future__ import annotations

from dataclasses import dataclass
from random import uniform

import pygame

from config import FOOD_COLOR, FOOD_RADIUS, WINDOW_HEIGHT, WINDOW_WIDTH


@dataclass
class Food:
    x: float
    y: float
    radius: float = FOOD_RADIUS

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.circle(
            surface,
            FOOD_COLOR,
            (int(self.x), int(self.y)),
            int(self.radius),
        )


def spawn_food_random(margin: int = 20) -> Food:
    """Spawn food at a random position inside the world, away from borders."""
    x = uniform(margin, WINDOW_WIDTH - margin)
    y = uniform(margin, WINDOW_HEIGHT - margin)
    return Food(x=x, y=y)

