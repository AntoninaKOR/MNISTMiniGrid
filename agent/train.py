"""Train a recurrent PPO agent on :class:`env.MNISTMazeVecEnv`.

The pipeline is:

1. Load a pre-trained MNIST classifier from disk (see
   :mod:`agent.pretrain_mnist`). It is used as a frozen observation encoder:
   the policy never sees raw pixels, only the predicted digit class.
2. Build a vectorised maze of the requested ``--size`` with a fresh random
   layout (obstacles + coloring) and a :class:`agent.policy.GRUPolicy`.
3. Run PPO + GAE.

One run trains a single board size; use ``--size 10 / 20 / 30`` for the three
requested scales.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from agent.mnist_classifier import load_classifier
from agent.policy import GRUPolicy
from agent.ppo import PPOConfig, collect_rollout, ppo_update
from env import MNISTMazeVecEnv, load_mnist_by_class, random_color_map, random_obstacle_mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=10, help="Maze side length (10, 20, or 30).")
    p.add_argument("--n-colors", type=int, default=10)
    p.add_argument("--obstacle-fraction", type=float, default=0.12)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--rollout-length", type=int, default=128)
    p.add_argument("--total-steps", type=int, default=300_000, help="Total environment steps.")
    p.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Defaults to 10 * size.",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument(
        "--value-clip-eps",
        type=float,
        default=-1.0,
        help="Per-epoch value-clip range. Disabled by default.",
    )
    p.add_argument(
        "--lr-anneal",
        action="store_true",
        default=False,
        help="Linearly decay lr from its initial value to 0 over training.",
    )
    p.add_argument(
        "--clip-anneal",
        action="store_true",
        default=False,
        help="Linearly decay clip_eps from its initial value to clip-anneal-min.",
    )
    p.add_argument("--clip-anneal-min", type=float, default=0.05)
    p.add_argument(
        "--entropy-anneal",
        action="store_true",
        default=False,
        help="Linearly decay entropy_coef from its initial value to --entropy-anneal-min.",
    )
    p.add_argument("--entropy-anneal-min", type=float, default=0.005)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument(
        "--action-embed-dim",
        type=int,
        default=32,
        help="Embedding size for prev_action. 0 = remove prev_action from GRU input entirely.",
    )
    p.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of stacked GRU layers.",
    )
    p.add_argument(
        "--curriculum-start",
        type=int,
        default=2,
        help="Initial max Manhattan distance from start to goal at episode reset.",
    )
    p.add_argument(
        "--curriculum-end",
        type=int,
        default=None,
        help="Final max Manhattan distance; defaults to the full board diameter (height + width).",
    )
    p.add_argument(
        "--curriculum-fraction",
        type=float,
        default=0.5,
        help="Fraction of --total-steps over which max_goal_distance grows linearly "
        "from --curriculum-start to --curriculum-end. After that it stays at the end value.",
    )
    p.add_argument(
        "--adaptive-curriculum",
        action="store_true",
        default=False,
        help="When set, advance d_max only when rolling success rate exceeds "
        "--curriculum-threshold (ignores --curriculum-fraction).",
    )
    p.add_argument(
        "--curriculum-threshold",
        type=float,
        default=0.7,
        help="Success rate that triggers d_max advancement when --adaptive-curriculum is used.",
    )
    p.add_argument(
        "--curriculum-window",
        type=int,
        default=20,
        help="Number of recent rollouts used to estimate success rate for adaptive curriculum.",
    )
    p.add_argument(
        "--mnist-checkpoint",
        type=Path,
        default=Path("checkpoints/mnist_classifier.pt"),
    )
    p.add_argument("--mnist-cache", type=Path, default=Path(".mnist_cache"))
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the trained policy state_dict (defaults to checkpoints/policy_<size>x<size>.pt).",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=1, help="Print stats every N rollouts.")
    return p.parse_args()


def curriculum_max_distance(step: int, total_steps: int, fraction: float, start: int, end: int) -> int:
    """Linear ramp from ``start`` to ``end`` over the first ``fraction`` of training."""
    ramp_steps = max(1, int(round(total_steps * fraction)))
    if step >= ramp_steps:
        return end
    t = step / ramp_steps
    return int(round(start + t * (end - start)))


def main() -> None:
    args = parse_args()
    assert args.num_envs % args.minibatches == 0
    height = width = args.size
    max_episode_steps = args.max_episode_steps or 10 * args.size
    diameter = height + width
    curriculum_end = args.curriculum_end if args.curriculum_end is not None else diameter

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # Layout (fixed for the whole run, per task spec).
    layout_rng = np.random.default_rng(args.seed)
    obstacles = random_obstacle_mask(height, width, args.obstacle_fraction, layout_rng)
    colors = random_color_map(height, width, args.n_colors, layout_rng)
    mnist_banks = load_mnist_by_class(cache_dir=args.mnist_cache)

    env = MNISTMazeVecEnv(
        num_envs=args.num_envs,
        height=height,
        width=width,
        obstacle_mask=obstacles,
        color_map=colors,
        n_colors=args.n_colors,
        max_steps=max_episode_steps,
        mnist_images_by_class=mnist_banks,
        seed=args.seed,
        max_goal_distance=args.curriculum_start,
    )

    classifier = load_classifier(args.mnist_checkpoint, device=device)
    policy = GRUPolicy(
        n_digits=10,
        goal_dim=height + width,
        n_actions=4,
        hidden_dim=args.hidden_dim,
        action_embed_dim=args.action_embed_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    cfg = PPOConfig(
        rollout_length=args.rollout_length,
        epochs=args.epochs,
        minibatches=args.minibatches,
        lr=args.lr,
        clip_eps=args.clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        value_clip_eps=(None if args.value_clip_eps < 0 else args.value_clip_eps),
    )

    obs, _ = env.reset(seed=args.seed)
    episode_start = np.ones(env.num_envs, dtype=bool)
    hidden = policy.initial_hidden(env.num_envs, device)
    prev_action = policy.initial_prev_action(env.num_envs, device)

    steps_per_rollout = args.rollout_length * args.num_envs
    n_rollouts = max(1, args.total_steps // steps_per_rollout)
    env_steps = 0
    t0 = time.time()

    # Rolling window for adaptive curriculum.
    succ_window: list[float] = []

    metric_keys = [
        "env_steps",
        "d_max",
        "lr_now",
        "clip_eps_now",
        # Rollout / episode stats
        "ep_return_mean",
        "ep_length_mean",
        "ep_success_rate",
        "n_episodes",
        "ep_truncated_frac",
        "term_value_mean",
        "reward_density",
        "observed_action_entropy",
        # PPO update stats
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_frac",
        "grad_norm",
        "explained_var",
        "value_mean",
        "value_std",
        "returns_mean",
        "returns_std",
        "adv_std_raw",
    ]
    metrics_log: list[dict[str, float]] = []

    anneal_mode = args.lr_anneal or args.clip_anneal or args.entropy_anneal
    print(
        f"Starting training: size={args.size}x{args.size}  "
        f"num_envs={args.num_envs}  rollout_len={args.rollout_length}  "
        f"total_steps={args.total_steps}  max_ep_steps={max_episode_steps}  device={device}"
    )
    if args.adaptive_curriculum:
        print(
            f"Curriculum: adaptive, d_max starts at {args.curriculum_start}, "
            f"advances when succ(window={args.curriculum_window}) > {args.curriculum_threshold}."
        )
    else:
        print(
            f"Curriculum: linear ramp {args.curriculum_start} -> {curriculum_end} "
            f"over first {int(args.curriculum_fraction * 100)}% of training."
        )
    if anneal_mode:
        print(f"Annealing: lr={'ON' if args.lr_anneal else 'OFF'}  clip={'ON' if args.clip_anneal else 'OFF'}  entropy={'ON' if args.entropy_anneal else 'OFF'}")

    for rollout_idx in range(n_rollouts):
        # ---- Annealing ----
        frac_remaining = 1.0 - env_steps / max(1, args.total_steps)
        lr_now = args.lr * frac_remaining if args.lr_anneal else args.lr
        clip_now = (
            max(args.clip_anneal_min, args.clip_eps * frac_remaining)
            if args.clip_anneal
            else args.clip_eps
        )
        entropy_now = (
            max(args.entropy_anneal_min, args.entropy_coef * frac_remaining)
            if args.entropy_anneal
            else args.entropy_coef
        )
        if anneal_mode:
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now
            cfg.clip_eps = clip_now
            cfg.entropy_coef = entropy_now

        # ---- Curriculum ----
        # Linear floor guarantees d_max always advances (avoids permanent stall).
        linear_d = curriculum_max_distance(
            env_steps,
            args.total_steps,
            args.curriculum_fraction,
            args.curriculum_start,
            curriculum_end,
        )
        if args.adaptive_curriculum:
            # Adaptive boost: advance one extra step when rolling succ > threshold.
            # We keep the window rolling (no clear) so context survives an advance.
            current_d = env.max_goal_distance or args.curriculum_start
            if (
                len(succ_window) >= args.curriculum_window
                and float(np.mean(succ_window)) >= args.curriculum_threshold
                and current_d < curriculum_end
            ):
                current_d = min(current_d + 1, curriculum_end)
            # Floor: never go below what the linear schedule requires.
            env.max_goal_distance = max(current_d, linear_d)
        else:
            env.max_goal_distance = linear_d

        rollout, hidden, prev_action, obs, episode_start, info = collect_rollout(
            env, classifier, policy,
            rollout_length=cfg.rollout_length,
            hidden=hidden,
            prev_action=prev_action,
            last_obs=obs,
            last_episode_start=episode_start,
            device=device,
        )
        update_stats = ppo_update(policy, optimizer, rollout, cfg)
        env_steps += steps_per_rollout

        if not np.isnan(info["ep_success_rate"]):
            succ_window.append(info["ep_success_rate"])
            if len(succ_window) > args.curriculum_window:
                succ_window.pop(0)

        metrics_log.append({
            "env_steps": env_steps,
            "d_max": env.max_goal_distance,
            "lr_now": lr_now,
            "clip_eps_now": clip_now,
            **{k: info[k] for k in (
                "ep_return_mean", "ep_length_mean", "ep_success_rate", "n_episodes",
                "ep_truncated_frac", "term_value_mean", "reward_density",
                "observed_action_entropy",
            )},
            **update_stats,
        })

        if (rollout_idx + 1) % args.log_every == 0 or rollout_idx == n_rollouts - 1:
            sps = env_steps / max(1e-6, time.time() - t0)
            print(
                f"[{env_steps:>8d}/{args.total_steps}]  "
                f"d_max={env.max_goal_distance:>3d}  "
                f"succ={info['ep_success_rate']:.2f}  "
                f"ep_len={info['ep_length_mean']:.1f}  "
                f"n_ep={info['n_episodes']:>3d}  "
                f"trunc={info['ep_truncated_frac']:.2f}  "
                f"Vmean={update_stats['value_mean']:+.2f}  "
                f"EV={update_stats['explained_var']:+.2f}  "
                f"pi={update_stats['policy_loss']:+.3f}  "
                f"v={update_stats['value_loss']:.3f}  "
                f"H={update_stats['entropy']:.2f}  "
                f"Hact={info['observed_action_entropy']:.2f}  "
                f"kl={update_stats['approx_kl']:+.3f}  "
                f"|g|={update_stats['grad_norm']:.2f}  "
                f"lr={lr_now:.2e}  "
                f"H_coef={entropy_now:.4f}  "
                f"sps={sps:.0f}"
            )

    out = args.output or Path(f"checkpoints/policy_{args.size}x{args.size}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": policy.state_dict(),
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "color_map": colors,
            "obstacle_mask": obstacles,
        },
        out,
    )

    metrics_path = out.with_name(out.stem + "_metrics.csv")
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metric_keys)
        writer.writeheader()
        writer.writerows(metrics_log)

    print(f"Saved policy to   {out}")
    print(f"Saved metrics to  {metrics_path}")


if __name__ == "__main__":
    main()
