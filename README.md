# MNIST MiniGrid

A vectorized, GridWorld-style maze environment whose observations are MNIST
handwritten digit images. Each cell has a *color* (a digit class); on every
step the agent sees one MNIST sample drawn from the class of the cell it
currently perceives.

![20x20 random rollout](docs/rollout_20x20_n10.gif)

## Environment

* Rectangular play area of size `(height, width)` surrounded by walls.
* Obstacles inside the area are specified by a binary `(height, width)` mask.
* Cells fall into three types: **floor**, **wall**, **obstacle**.
* There are `n_colors ≤ 10` colors total. Walls and obstacles each have one
  dedicated color; floor cells use the remaining `n_colors - 2` colors.
* The coloring of the internal area is passed in at construction time and
  stays fixed throughout training.

When `n_colors = 3` only a single floor color is left, so every walkable cell
looks the same and the MNIST digit class the agent sees on the floor never
changes between cells:

![n_colors=3 demo](docs/rollout_10x10_n3.gif)

### Actions

Four discrete actions, numbered clockwise starting from "up":

| id | name  | (Δrow, Δcol) |
|----|-------|--------------|
| 0  | up    | (-1,  0)     |
| 1  | right | ( 0, +1)     |
| 2  | down  | (+1,  0)     |
| 3  | left  | ( 0, -1)     |

If the move targets a wall (outside the area) or an obstacle, the agent
**stays in place** but its observation is taken from the blocked cell — so it
*sees* the wall or obstacle. Below, the agent is placed at the top edge and
keeps sending action `0` ("up"); it never moves, and the MNIST panel cycles
through samples of the wall class (digit `0`, since `wall_color = 0`):

![bumping into a wall](docs/rollout_10x10_wall.gif)

### Observation

Each step returns a dict:

```python
{
  "image": np.uint8[28, 28],                  # MNIST sample of the perceived cell
  "goal":  np.float32[height + width],        # two-hot encoding of the goal (row, col)
}
```

The `image` for each color index is sampled uniformly from the MNIST training
images of the corresponding digit class. The `goal` is a standard two-hot
encoding: a 1 at the goal's row index in the first `height` slots, and a 1 at
its column index in the last `width` slots.

Because the image is resampled at every step, even an agent that stays at the
exact same cell keeps seeing different MNIST samples of the *same* digit
class — the class index stays constant while the handwriting changes:

![agent stays in place, observation keeps changing](docs/rollout_10x10_stay.gif)

### Reward & termination

* Reward is sparse: `+1` on the step that reaches the goal, `0` otherwise.
* The episode **terminates** when the agent reaches the goal.
* The episode **truncates** at `max_steps`.

## Vectorization

`MNISTMazeVecEnv` subclasses `gymnasium.vector.VectorEnv` and runs
`num_envs` sub-environments in lockstep. It exposes the standard
`single_observation_space`, `single_action_space`, `observation_space`, and
`action_space` attributes. Autoreset uses `NEXT_STEP` semantics: a
sub-environment that terminates or truncates on step `t` is automatically reset
at the start of step `t + 1` (its action is ignored that step, the returned
observation is the reset observation, and `reward = 0`, `terminated =
truncated = False`).

## Installation

Requirements: Python ≥ 3.10. 

```bash
pip install -e .  
```

The first time `MNISTMazeVecEnv` is constructed it downloads the MNIST
training files (~11 MB) into `~/.cache/mnist-maze/`. To use a different cache
location, pass it explicitly:

```python
from env import load_mnist_by_class, MNISTMazeVecEnv

mnist_banks = load_mnist_by_class(cache_dir="./.mnist_cache")
env = MNISTMazeVecEnv(..., mnist_images_by_class=mnist_banks)
```

The loader needs just two files —
`train-images-idx3-ubyte.gz` and `train-labels-idx1-ubyte.gz`. If they are
already present in the cache directory, no network access is required.

## Usage

```python
import numpy as np
from env import (
    MNISTMazeVecEnv,
    random_color_map,
    random_obstacle_mask,
)

rng = np.random.default_rng(0)
height, width = 20, 20
obstacles = random_obstacle_mask(height, width, fraction=0.12, rng=rng)
colors = random_color_map(height, width, n_colors=10, rng=rng)

env = MNISTMazeVecEnv(
    num_envs=8,
    height=height,
    width=width,
    obstacle_mask=obstacles,
    color_map=colors,
    n_colors=10,
    max_steps=4 * height * width,
    seed=0,
)

obs, info = env.reset(seed=0)
for _ in range(1000):
    actions = rng.integers(0, 4, size=env.num_envs)
    obs, reward, terminated, truncated, info = env.step(actions)
```

## Rendering

`MNISTMazeVecEnv.render_frame(env_idx=0, cell_size=24)` returns a
`(H, W, 3) uint8` RGB frame for one sub-environment. The layout is:

* **Left**: the MNIST image the agent is currently observing (scaled up).
* **Centre**: a *color → digit* legend showing which palette entry corresponds
  to each MNIST class (the wall and obstacle color indices are included).
* **Right**: the colored maze with a one-cell-thick dark slate wall border,
  obstacles (color index 1, near-black), the **start cell** of the current
  episode (green hollow square), the **goal** (yellow hollow square), and the
  **agent** (white circle).
* **Bottom**: a status bar with the current step count and the episode score
  (cumulative reward of the current episode, tracked in
  `env.episode_return[env_idx]`).

## Training a PPO agent

The `agent/` package contains the training pipeline: a pre-trained MNIST
classifier (used as a frozen observation encoder), a recurrent (GRU)
actor-critic policy, PPO + GAE.

### Architecture

* **Observation encoder** — a small CNN (`agent.mnist_classifier.MNISTClassifier`)
  is pre-trained on MNIST and **frozen**. At every step the agent's raw
  `28 × 28` MNIST observation is converted to a single predicted digit
  (`argmax` of the classifier logits). The policy therefore sees the *digit
  class* (0–9), not the pixels.
* **Policy** — `agent.policy.GRUPolicy`: embed the digit (`Embedding(10, 32)`),
  project the two-hot goal (`Linear(h + w, 32)`), concatenate, feed into a
  `GRUCell(64, hidden=128)`, then two linear heads. Hidden state is reset at every episode boundary.
* **Algorithm** — `agent.ppo`: recurrent PPO with GAE
  (γ=0.99, λ=0.95, ε=0.2), advantages normalised globally per rollout,
  minibatches over *envs* so the time axis stays intact for the GRU.

### Why this design

* **Frozen MNIST classifier as the observation encoder.** The cell observation
  is intentionally a noisy view of a single discrete signal — the cell's
  *color index*, which is also the MNIST digit class. This decouples representation learning from RL credit
  assignment, makes the policy small and fast, and avoids the well-known
  sample-inefficiency of pixel-based RL on a sparse-reward task.

* **Recurrent (GRU) policy.** The env is strongly partially observable —
  one cell per step, no compass, no map. The agent has to **integrate over
  time** the digits it has seen, the wall/obstacle bumps, and the actions it
  took in order to localise itself relative to the (known) goal coordinates.
  A GRU is the smallest standard recurrent block that handles this.
* **PPO + GAE.** PPO is the standard robust on-policy choice and works well
  with recurrent networks. The clipped surrogate objective keeps updates
  stable without explicit trust-region machinery, and GAE turns the very
  sparse reward (one ±1 spike per episode) into a smoothly bootstrapped
  advantage signal that PPO can actually learn from.

### Pre-train the MNIST classifier (once)

```bash
python -m agent.pretrain_mnist --epochs 3 \
    --output checkpoints/mnist_classifier.pt
```

### Train PPO on a single board size

One run trains one size; use `--size 10 / 20 / 30` for the three requested
scales.

```bash
python -m agent.train --size 10 --total-steps 1_000_000
python -m agent.train --size 20 --total-steps 2_000_000
python -m agent.train --size 30 --total-steps 4_000_000
```

The obstacle mask and color map are sampled once at startup from `--seed`
and stay fixed for the entire run. Each run produces two artefacts in
`checkpoints/`:

* `policy_<H>x<W>.pt` — policy weights + the layout + CLI arguments;
* `policy_<H>x<W>_metrics.csv` — per-rollout training metrics, consumed by
  `agent.eval` to plot learning curves.

Key flags (see `--help` for the rest):

| flag | default | meaning |
|------|---------|---------|
| `--size` | `10` | side length of the square maze |
| `--num-envs` | `16` | parallel sub-environments (must divide `--minibatches`) |
| `--rollout-length` | `128` | env steps per PPO rollout |
| `--total-steps` | `200_000` | total environment steps |
| `--max-episode-steps` | `4 * size` | episode length limit |
| `--lr` | `3e-4` | Adam learning rate |
| `--gamma` / `--gae-lambda` | `0.99 / 0.95` | discount and GAE λ |
| `--mnist-checkpoint` | `checkpoints/mnist_classifier.pt` | frozen encoder weights |
| `--device` | `cpu` | use `cuda` / `mps` if available |

Per-rollout metrics print as:

```
[  16384/200000]  ep_ret=0.84  ep_len=12.3  succ=0.84  n_ep= 67  pi_loss=-0.011  v_loss=0.041  H=1.105  kl=+0.005  sps=1490
```

`ep_ret` is the mean episode return for episodes that finished within the
rollout, `succ` is the share of them that actually reached the goal, and
`sps` is environment steps per wall-clock second.

### Evaluate a trained policy: learning curves + GIF

`agent.eval` takes a policy checkpoint, plots the recorded learning curves
and records a GIF of the trained agent acting in the env:

```bash
python -m agent.eval --policy checkpoints/policy_10x10.pt
```

**Validation runs on the same environment the agent was trained on.** The
training checkpoint stores the original `obstacle_mask` and `color_map`
together with the relevant CLI arguments (`size`, `n_colors`,
`max_episode_steps`, `mnist_checkpoint`, `hidden_dim`), and `agent.eval`
rebuilds the env from those exact values. The only thing that differs from
training is the RNG seed (`--seed`, default `12345`), so that the recorded
episodes have fresh start/goal positions on the same fixed layout — i.e.
this is an *in-distribution* evaluation, not a generalisation test on unseen
mazes.

By default this writes, next to the policy file:

* `policy_<H>x<W>_curves.png` — 4-panel figure with mean episode return,
  mean episode length, success rate, and policy entropy + approx KL versus
  env steps (read straight from the metrics CSV).
* `policy_<H>x<W>_rollout.gif` — N completed episodes of the trained
  policy, rendered with the same MNIST-observation/legend/maze layout as the
  random-agent GIFs above.

Useful flags:

| flag | default | meaning |
|------|---------|---------|
| `--num-episodes` | `3` | how many completed episodes to capture in the GIF |
| `--max-frames` | `400` | hard cap on total GIF frames (safety against very long episodes) |
| `--deterministic` | off | greedy `argmax` actions; otherwise sample from the actor distribution |
| `--cell-size` | `28` | pixel size of one maze cell in the GIF |
| `--fps` | `3.0` | GIF playback speed |
| `--gif` / `--plot` / `--metrics` | derived from `--policy` | override output paths |
| `--device` | `cpu` | use `cuda` / `mps` if available |

