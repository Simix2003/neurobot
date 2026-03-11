from __future__ import annotations

"""Reward shaping utilities for NeuroBot RL training.

The goal is to be consistent with the episode-level metrics used in the
interactive simulation while providing a dense per-step signal:

- strong positive reward when food is collected
- small positive reward for moving closer to the food
- small penalties for movement and time to encourage efficiency
"""


def compute_step_reward(
    prev_food_distance: float,
    new_food_distance: float,
    step_distance: float,
    food_eaten: int,
) -> float:
    """Compute a single-step reward signal.

    Args:
        prev_food_distance: normalized [0, 1] distance to food before the step.
        new_food_distance: normalized [0, 1] distance to food after the step.
        step_distance: distance the robot moved in this step (pixels).
        food_eaten: how many food items were collected on this step (usually 0 or 1).
    """
    # Episode-level analogue:
    #   reward_episode = food_reward * food_collected - distance_penalty * distance_traveled
    FOOD_REWARD = 10.0
    MOVE_PENALTY = 0.01
    TIME_PENALTY = 0.001
    APPROACH_SCALE = 1.0

    reward = 0.0

    # Sparse event: eating food
    if food_eaten > 0:
        reward += FOOD_REWARD * float(food_eaten)

    # Dense signal: moving closer to the food between timesteps
    delta_food = prev_food_distance - new_food_distance
    reward += APPROACH_SCALE * delta_food

    # Penalize movement (encourages shorter paths) and a tiny time cost
    reward -= MOVE_PENALTY * step_distance
    reward -= TIME_PENALTY

    return float(reward)

