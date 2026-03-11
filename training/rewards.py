from __future__ import annotations

"""Reward shaping utilities for NeuroBot RL training.

The goal is to be consistent with the episode-level metrics used in the
interactive simulation while providing a dense per-step signal:

- strong positive reward when food is collected
- strong positive reward for moving closer to the food
- clear penalties for moving away from food or toward walls
- gentle penalties for movement and time to encourage efficiency
"""


# Tunable global reward constants
FOOD_REWARD: float = 80.0
APPROACH_SCALE: float = 4.0
MOVE_PENALTY: float = 0.001
TIME_PENALTY: float = 0.0001

WALL_FRONT_PENALTY: float = 1.0
WALL_SIDE_PENALTY: float = 0.3
WALL_THRESHOLD: float = 0.1


def compute_step_reward(
    prev_food_distance: float,
    new_food_distance: float,
    step_distance: float,
    food_eaten: int,
    wall_front: float,
    wall_left: float,
    wall_right: float,
) -> float:
    """Compute a single-step reward signal.

    Args:
        prev_food_distance: normalized [0, 1] distance to food before the step.
        new_food_distance: normalized [0, 1] distance to food after the step.
        step_distance: distance the robot moved in this step (pixels).
        food_eaten: how many food items were collected on this step (usually 0 or 1).
        wall_front: normalized distance to wall in front.
        wall_left: normalized distance to wall on the left.
        wall_right: normalized distance to wall on the right.
    """
    reward = 0.0

    # Sparse event: eating food
    if food_eaten > 0:
        reward += FOOD_REWARD * float(food_eaten)

    # Dense signal: moving closer to the food between timesteps
    delta_food = prev_food_distance - new_food_distance
    if delta_food > 0.0:
        # Reward moving closer
        reward += APPROACH_SCALE * delta_food
    elif delta_food < 0.0:
        # Penalize moving away slightly more strongly
        reward += 1.5 * APPROACH_SCALE * delta_food

    # Penalize being very close to walls, especially in front
    if wall_front < WALL_THRESHOLD:
        # Scale penalty by how deep into the danger zone we are
        factor = (WALL_THRESHOLD - wall_front) / WALL_THRESHOLD
        reward -= WALL_FRONT_PENALTY * factor

    for side_dist in (wall_left, wall_right):
        if side_dist < WALL_THRESHOLD:
            factor = (WALL_THRESHOLD - side_dist) / WALL_THRESHOLD
            reward -= WALL_SIDE_PENALTY * factor

    # Penalize movement (encourages shorter paths) and a tiny time cost
    reward -= MOVE_PENALTY * step_distance
    reward -= TIME_PENALTY

    return float(reward)

