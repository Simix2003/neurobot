from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from training.rl_env import NeuroBotRLEnv
from training.policy import PolicyNetwork
from training.io import ensure_dir, save_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NeuroBot with REINFORCE.")
    parser.add_argument(
        "--episodes",
        type=int,
        required=True,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=30.0,
        help="Target duration of each episode in seconds (default: 30).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor for returns (default: 0.95).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for the policy optimizer (default: 3e-4).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save a policy checkpoint every N episodes (default: 50).",
    )
    parser.add_argument(
        "--batch-episodes",
        type=int,
        default=5,
        help="Number of episodes per policy update (default: 5).",
    )
    return parser.parse_args()


def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    """Compute discounted returns G_t from a list of rewards."""
    returns: List[float] = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.append(g)
    returns.reverse()
    returns_t = torch.tensor(returns, dtype=torch.float32)
    # Normalize for stability
    if returns_t.std() > 1e-8:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
    return returns_t


def main() -> None:
    args = parse_args()

    # Reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    base_dir = os.path.dirname(__file__)
    runs_dir = os.path.join(base_dir, "..", "runs")
    runs_dir = os.path.normpath(runs_dir)
    ensure_dir(runs_dir)

    train_log_path = os.path.join(runs_dir, "train_log.csv")
    policies_dir = os.path.join(runs_dir, "policies")
    ensure_dir(policies_dir)

    # Prepare train_log.csv with header if needed
    if not os.path.exists(train_log_path) or os.path.getsize(train_log_path) == 0:
        with open(train_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "return",
                    "loss",
                    "mean_reward",
                    "length_steps",
                    "best_return_so_far",
                    "food_collected",
                    "distance_traveled",
                ]
            )

    env = NeuroBotRLEnv(episode_seconds=args.episode_seconds)
    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    best_return = float("-inf")

    ep = 1
    while ep <= args.episodes:
        batch_log_probs: List[torch.Tensor] = []
        batch_advantages: List[torch.Tensor] = []
        batch_values: List[torch.Tensor] = []
        batch_returns: List[torch.Tensor] = []

        episodes_this_batch = min(args.batch_episodes, args.episodes - ep + 1)

        for _ in range(episodes_this_batch):
            obs_np = env.reset()
            log_probs: List[torch.Tensor] = []
            rewards: List[float] = []
            values: List[torch.Tensor] = []

            done = False
            steps = 0
            last_info: Dict[str, Any] = {}

            while not done:
                obs_t = torch.from_numpy(obs_np.astype(np.float32))
                action_t, log_prob_t, value_t = policy.act(obs_t, deterministic=False)

                next_obs_np, reward, done, info = env.step(action_t.detach().numpy())

                log_probs.append(log_prob_t)
                rewards.append(float(reward))
                values.append(value_t)

                obs_np = next_obs_np
                steps += 1
                last_info = info

            returns_t = compute_returns(rewards, args.gamma)
            values_t = torch.stack(values)
            log_probs_t = torch.stack(log_probs)

            advantages_t = returns_t - values_t.detach()

            batch_log_probs.append(log_probs_t)
            batch_returns.append(returns_t)
            batch_values.append(values_t)
            batch_advantages.append(advantages_t)

            episode_return = float(returns_t[0].item())
            mean_reward = float(np.mean(rewards)) if rewards else 0.0
            best_return = max(best_return, episode_return)

            # Environment-based metrics (if available)
            food_collected = 0
            distance_traveled = 0.0
            episode_metrics = last_info.get("episode_metrics") if last_info else None
            if isinstance(episode_metrics, dict):
                food_collected = int(episode_metrics.get("food_collected", 0))
                distance_traveled = float(episode_metrics.get("distance_traveled", 0.0))

            # We log per-episode using the current (batched) loss placeholder; updated after batch update
            print(
                f"Ep {ep}/{args.episodes} | "
                f"return={episode_return:.3f} | "
                f"mean_r={mean_reward:.3f} | "
                f"len={steps} | "
                f"best_return={best_return:.3f} | "
                f"food={food_collected} | "
                f"dist={distance_traveled:.1f}"
            )

            # Temporarily store metrics to write after loss is computed
            with open(train_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        ep,
                        episode_return,
                        0.0,  # placeholder loss, will be approximated by batch loss
                        mean_reward,
                        steps,
                        best_return,
                        food_collected,
                        distance_traveled,
                    ]
                )

            ep += 1
            if ep > args.episodes:
                break

        # Batch update with advantage-based loss and gradient clipping
        all_log_probs = torch.cat(batch_log_probs)
        all_returns = torch.cat(batch_returns)
        all_values = torch.cat(batch_values)
        all_advantages = torch.cat(batch_advantages)

        policy_loss = -(all_log_probs * all_advantages).sum()
        value_loss = F.mse_loss(all_values, all_returns)
        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        # Periodic checkpointing (after each batch)
        last_ep_in_batch = min(ep - 1, args.episodes)
        if last_ep_in_batch % args.checkpoint_every == 0 or last_ep_in_batch == args.episodes:
            ckpt_path = os.path.join(policies_dir, f"policy_ep{last_ep_in_batch:04d}.pt")
            save_policy(policy, ckpt_path)


if __name__ == "__main__":
    main()

