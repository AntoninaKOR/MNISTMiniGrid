"""Recurrent PPO with Generalized Advantage Estimation.

Designed for :class:`env.MNISTMazeVecEnv` (a vectorised env with NEXT_STEP
autoreset) and :class:`agent.policy.GRUPolicy` (a GRUCell-based actor-critic
that takes pre-classified digit observations).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from agent.mnist_classifier import MNISTClassifier
from agent.policy import GRUPolicy
from env import MNISTMazeVecEnv


@dataclass
class PPOConfig:
    rollout_length: int = 128
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    value_clip_eps: float | None = None


@dataclass
class Rollout:
    digits: torch.Tensor          # (T, B) long
    goals: torch.Tensor           # (T, B, goal_dim) float
    prev_actions: torch.Tensor    # (T, B) long; n_actions = "no previous action" token
    actions: torch.Tensor         # (T, B) long
    log_probs: torch.Tensor       # (T, B) float
    values: torch.Tensor          # (T, B) float
    rewards: torch.Tensor         # (T, B) float
    dones: torch.Tensor           # (T, B) bool (term | trunc at step t)
    episode_starts: torch.Tensor  # (T, B) bool (first step after a reset)
    initial_hidden: torch.Tensor  # (num_layers, B, hidden)
    last_value: torch.Tensor      # (B,)


@torch.no_grad()
def collect_rollout(
    env: MNISTMazeVecEnv,
    classifier: MNISTClassifier,
    policy: GRUPolicy,
    *,
    rollout_length: int,
    hidden: torch.Tensor,
    prev_action: torch.Tensor,
    last_obs: dict[str, np.ndarray],
    last_episode_start: np.ndarray,
    device: torch.device,
) -> tuple[
    Rollout,
    torch.Tensor,
    torch.Tensor,
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, float],
]:
    """Step the env ``rollout_length`` times and pack the trajectory.

    Returns ``(rollout, new_hidden, new_prev_action, last_obs, last_episode_start, info)``
    so the next rollout can pick up from where this one left off. ``info``
    contains aggregate statistics over the rollout.
    """
    T = rollout_length
    B = env.num_envs
    goal_dim = env.height + env.width

    digits_buf = torch.empty(T, B, dtype=torch.long, device=device)
    goals_buf = torch.empty(T, B, goal_dim, dtype=torch.float32, device=device)
    prev_actions_buf = torch.empty(T, B, dtype=torch.long, device=device)
    actions_buf = torch.empty(T, B, dtype=torch.long, device=device)
    log_probs_buf = torch.empty(T, B, dtype=torch.float32, device=device)
    values_buf = torch.empty(T, B, dtype=torch.float32, device=device)
    rewards_buf = torch.empty(T, B, dtype=torch.float32, device=device)
    dones_buf = torch.empty(T, B, dtype=torch.bool, device=device)
    starts_buf = torch.empty(T, B, dtype=torch.bool, device=device)

    initial_hidden = hidden.clone()

    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    completed_success: list[float] = []

    null_action = policy.initial_prev_action(B, device)

    for t in range(T):
        image_np = last_obs["image"]
        goal_np = last_obs["goal"]
        image_t = torch.from_numpy(image_np).to(device)
        goal_t = torch.from_numpy(goal_np).to(device)
        episode_start_t = torch.from_numpy(last_episode_start).to(device)

        # Reset hidden state and prev_action where the previous step ended an
        # episode -- the agent at a freshly-reset env has no "previous action".
        # ``hidden`` has shape (num_layers, B, H); broadcast mask over layers.
        hidden = hidden * (~episode_start_t).float().view(1, -1, 1)
        prev_action = torch.where(episode_start_t, null_action, prev_action)

        digit = classifier.predict(image_t)
        logits, value, hidden = policy.step(digit, goal_t, prev_action, hidden)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        digits_buf[t] = digit
        goals_buf[t] = goal_t
        prev_actions_buf[t] = prev_action
        actions_buf[t] = action
        log_probs_buf[t] = log_prob
        values_buf[t] = value
        starts_buf[t] = episode_start_t

        action_np = action.cpu().numpy()
        new_obs, reward, terminated, truncated, _ = env.step(action_np)
        done = np.logical_or(terminated, truncated)
        rewards_buf[t] = torch.from_numpy(reward.astype(np.float32)).to(device)
        dones_buf[t] = torch.from_numpy(done).to(device)

        # Episode statistics for envs that just finished. ``env.episode_return``
        # and ``env.step_count`` are updated inside ``step()``, and only reset
        # on the next call when NEXT_STEP autoreset kicks in -- so we read
        # them here before stepping again.
        for i in np.flatnonzero(done):
            completed_returns.append(float(env.episode_return[i]))
            completed_lengths.append(int(env.step_count[i]))
            completed_success.append(1.0 if terminated[i] else 0.0)

        prev_action = action
        last_obs = new_obs
        last_episode_start = done

    # Bootstrap value for the step right after the rollout.
    image_t = torch.from_numpy(last_obs["image"]).to(device)
    goal_t = torch.from_numpy(last_obs["goal"]).to(device)
    episode_start_t = torch.from_numpy(last_episode_start).to(device)
    next_hidden = hidden * (~episode_start_t).float().view(1, -1, 1)
    next_prev_action = torch.where(episode_start_t, null_action, prev_action)
    digit = classifier.predict(image_t)
    _, last_value, _ = policy.step(digit, goal_t, next_prev_action, next_hidden)

    rollout = Rollout(
        digits=digits_buf,
        goals=goals_buf,
        prev_actions=prev_actions_buf,
        actions=actions_buf,
        log_probs=log_probs_buf,
        values=values_buf,
        rewards=rewards_buf,
        dones=dones_buf,
        episode_starts=starts_buf,
        initial_hidden=initial_hidden,
        last_value=last_value,
    )
    info = {
        "ep_return_mean": float(np.mean(completed_returns)) if completed_returns else float("nan"),
        "ep_length_mean": float(np.mean(completed_lengths)) if completed_lengths else float("nan"),
        "ep_success_rate": float(np.mean(completed_success)) if completed_success else float("nan"),
        "n_episodes": len(completed_returns),
    }
    return rollout, hidden, prev_action, last_obs, last_episode_start, info


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard truncated GAE.

    ``dones[t]`` zeroes the bootstrap for step ``t`` (episode boundary).
    Returns ``(advantages, returns)`` of shape ``(T, B)``.
    """
    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)
    next_value = last_value
    for t in reversed(range(T)):
        not_done = (~dones[t]).float()
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_gae = delta + gamma * gae_lambda * not_done * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def ppo_update(
    policy: GRUPolicy,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    config: PPOConfig,
) -> dict[str, float]:
    """Run ``epochs`` * ``minibatches`` PPO updates over the rollout.

    Minibatches are formed by partitioning *envs* (not timesteps), so each
    minibatch keeps the full ``T``-step time axis intact for the GRU.
    """
    advantages, returns = compute_gae(
        rollout.rewards,
        rollout.values,
        rollout.dones,
        rollout.last_value,
        config.gamma,
        config.gae_lambda,
    )
    # Normalise advantages globally for stability.
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    B = rollout.digits.shape[1]
    assert B % config.minibatches == 0, "num_envs must be divisible by minibatches"
    mb_size = B // config.minibatches
    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_frac": 0.0,
        "grad_norm": 0.0,
    }
    n_updates = 0

    for _ in range(config.epochs):
        env_perm = torch.randperm(B, device=rollout.digits.device)
        for start in range(0, B, mb_size):
            mb_idx = env_perm[start : start + mb_size]
            new_log_probs, new_values, new_entropies = policy.evaluate(
                rollout.digits[:, mb_idx],
                rollout.goals[:, mb_idx],
                rollout.prev_actions[:, mb_idx],
                rollout.actions[:, mb_idx],
                rollout.episode_starts[:, mb_idx],
                rollout.initial_hidden[:, mb_idx],
            )
            old_log_probs = rollout.log_probs[:, mb_idx]
            old_values = rollout.values[:, mb_idx]
            mb_adv = adv_norm[:, mb_idx]
            mb_ret = returns[:, mb_idx]

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * mb_adv
            surr2 = ratio.clamp(1.0 - config.clip_eps, 1.0 + config.clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            if config.value_clip_eps is None:
                value_loss = 0.5 * (new_values - mb_ret).pow(2).mean()
            else:
                v_clipped = old_values + (new_values - old_values).clamp(
                    -config.value_clip_eps, config.value_clip_eps
                )
                v_loss_unclipped = (new_values - mb_ret).pow(2)
                v_loss_clipped = (v_clipped - mb_ret).pow(2)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy = new_entropies.mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_log_probs - new_log_probs).mean().item()
                clip_frac = ((ratio - 1.0).abs() > config.clip_eps).float().mean().item()
            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()
            stats["approx_kl"] += approx_kl
            stats["clip_frac"] += clip_frac
            stats["grad_norm"] += float(grad_norm)
            n_updates += 1

    for k in stats:
        stats[k] /= max(1, n_updates)
    return stats
