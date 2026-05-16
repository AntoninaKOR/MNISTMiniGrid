"""Smoke tests for the vectorized MNIST maze environment."""

from __future__ import annotations

import numpy as np
import pytest

from env import (
    MNISTMazeVecEnv,
    load_mnist_by_class,
    random_color_map,
    random_obstacle_mask,
    two_hot_encode,
)

# A small synthetic MNIST stand-in keeps the tests offline and fast.
# It has the same shape as the real dataset: 10 classes of (n_c, 28, 28) uint8.
SYNTHETIC_MNIST = [
    np.full((4, 28, 28), 10 * c, dtype=np.uint8) for c in range(10)
]


def _make_env(
    height: int,
    width: int,
    num_envs: int = 4,
    obstacle_fraction: float = 0.12,
    n_colors: int = 10,
    max_steps: int | None = None,
    seed: int = 0,
    mnist_images_by_class=SYNTHETIC_MNIST,
) -> MNISTMazeVecEnv:
    rng = np.random.default_rng(seed)
    obstacles = random_obstacle_mask(height, width, obstacle_fraction, rng)
    colors = random_color_map(height, width, n_colors, rng)
    return MNISTMazeVecEnv(
        num_envs=num_envs,
        height=height,
        width=width,
        obstacle_mask=obstacles,
        color_map=colors,
        n_colors=n_colors,
        max_steps=max_steps if max_steps is not None else 4 * height * width,
        seed=seed,
        mnist_images_by_class=mnist_images_by_class,
    )


@pytest.mark.parametrize("size", [10, 20, 30])
def test_basic_rollout(size: int) -> None:
    env = _make_env(size, size, num_envs=8)
    obs, info = env.reset(seed=123)

    assert obs["image"].shape == (8, 28, 28)
    assert obs["image"].dtype == np.uint8
    assert obs["goal"].shape == (8, size + size)
    assert obs["goal"].dtype == np.float32
    # Two-hot encoding has exactly two 1s per row.
    assert np.all(obs["goal"].sum(axis=1) == 2)

    rng = np.random.default_rng(0)
    for _ in range(50):
        actions = rng.integers(0, 4, size=env.num_envs)
        obs, rewards, terminated, truncated, info = env.step(actions)
        assert obs["image"].shape == (env.num_envs, 28, 28)
        assert rewards.shape == (env.num_envs,) and rewards.dtype == np.float32
        assert terminated.shape == (env.num_envs,) and terminated.dtype == bool
        assert truncated.shape == (env.num_envs,) and truncated.dtype == bool
        assert np.all((rewards == 0.0) | (rewards == 1.0))


def test_walls_and_obstacles_block_movement() -> None:
    # 3x3 maze with an obstacle at (1, 1); agent forced to start at (0, 0).
    obstacles = np.zeros((3, 3), dtype=bool)
    obstacles[1, 1] = True
    colors = np.full((3, 3), 2, dtype=np.int64)
    env = MNISTMazeVecEnv(
        num_envs=1,
        height=3,
        width=3,
        obstacle_mask=obstacles,
        color_map=colors,
        n_colors=10,
        max_steps=100,
        mnist_images_by_class=SYNTHETIC_MNIST,
        seed=0,
    )
    env.reset(seed=0)
    env.pos_agent[:] = np.array([[0, 0]])
    env.pos_goal[:] = np.array([[0, 2]])
    env.need_reset[:] = False
    env.step_count[:] = 0

    # Move up = action 0 -> wall, agent stays at (0, 0), sees wall color.
    _, reward, terminated, truncated, _ = env.step(np.array([0]))
    assert env.pos_agent.tolist() == [[0, 0]]
    assert env.last_color.tolist() == [env.wall_color]
    assert reward.tolist() == [0.0]
    assert not terminated.any() and not truncated.any()

    # Move down = action 2 -> floor cell (1, 0), agent moves, sees floor color.
    _, _, _, _, _ = env.step(np.array([2]))
    assert env.pos_agent.tolist() == [[1, 0]]
    assert env.last_color.tolist() == [int(colors[1, 0])]

    # Move right = action 1 -> obstacle at (1, 1), agent stays, sees obstacle color.
    _, _, _, _, _ = env.step(np.array([1]))
    assert env.pos_agent.tolist() == [[1, 0]]
    assert env.last_color.tolist() == [env.obstacle_color]


def test_goal_reached_gives_reward_and_autoreset() -> None:
    obstacles = np.zeros((3, 3), dtype=bool)
    colors = np.full((3, 3), 2, dtype=np.int64)
    env = MNISTMazeVecEnv(
        num_envs=1,
        height=3,
        width=3,
        obstacle_mask=obstacles,
        color_map=colors,
        n_colors=10,
        max_steps=100,
        mnist_images_by_class=SYNTHETIC_MNIST,
        seed=0,
    )
    env.reset(seed=0)
    env.pos_agent[:] = np.array([[0, 0]])
    env.pos_goal[:] = np.array([[0, 1]])
    env.need_reset[:] = False
    env.step_count[:] = 0

    # Step right -> reach goal.
    _, reward, terminated, truncated, _ = env.step(np.array([1]))
    assert reward.tolist() == [1.0]
    assert terminated.tolist() == [True]
    assert truncated.tolist() == [False]
    assert env.need_reset.tolist() == [True]

    # The very next step must auto-reset and ignore the action.
    prev_goal = env.pos_goal.copy()
    _, reward, terminated, truncated, _ = env.step(np.array([0]))
    assert reward.tolist() == [0.0]
    assert terminated.tolist() == [False]
    assert truncated.tolist() == [False]
    assert env.need_reset.tolist() == [False]
    assert env.step_count.tolist() == [0]
    # A fresh episode was sampled, so positions are most likely different.
    assert env.pos_goal.shape == prev_goal.shape


def test_truncation_at_max_steps() -> None:
    obstacles = np.zeros((3, 3), dtype=bool)
    colors = np.full((3, 3), 2, dtype=np.int64)
    env = MNISTMazeVecEnv(
        num_envs=1,
        height=3,
        width=3,
        obstacle_mask=obstacles,
        color_map=colors,
        n_colors=10,
        max_steps=2,
        mnist_images_by_class=SYNTHETIC_MNIST,
        seed=0,
    )
    env.reset(seed=0)
    env.pos_agent[:] = np.array([[0, 0]])
    env.pos_goal[:] = np.array([[2, 2]])  # unreachable in 2 wall-bumps
    env.need_reset[:] = False
    env.step_count[:] = 0

    # Two wall-bumps should trigger truncation on the second step.
    env.step(np.array([0]))
    _, reward, terminated, truncated, _ = env.step(np.array([0]))
    assert terminated.tolist() == [False]
    assert truncated.tolist() == [True]
    assert reward.tolist() == [0.0]


def test_helpers_are_deterministic() -> None:
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    mask_a = random_obstacle_mask(10, 10, 0.15, rng_a)
    mask_b = random_obstacle_mask(10, 10, 0.15, rng_b)
    assert np.array_equal(mask_a, mask_b)
    assert mask_a.sum() == int(round(10 * 10 * 0.15))

    cmap_a = random_color_map(10, 10, 10, rng_a)
    cmap_b = random_color_map(10, 10, 10, rng_b)
    assert np.array_equal(cmap_a, cmap_b)
    assert set(np.unique(cmap_a).tolist()).issubset(set(range(2, 10)))


def test_two_hot_encoding_shape() -> None:
    positions = np.array([[0, 0], [2, 4], [9, 9]])
    encoded = two_hot_encode(positions, 10, 10)
    assert encoded.shape == (3, 20)
    assert encoded.dtype == np.float32
    assert np.all(encoded.sum(axis=1) == 2)
    assert encoded[1, 2] == 1.0
    assert encoded[1, 10 + 4] == 1.0


def test_observation_action_spaces() -> None:
    env = _make_env(10, 10, num_envs=3)
    assert env.single_action_space.n == 4
    assert env.action_space.shape == (3,)
    assert "image" in env.single_observation_space.spaces
    assert "goal" in env.single_observation_space.spaces


def test_render_frame_shape_and_dtype() -> None:
    env = _make_env(10, 10, num_envs=2)
    env.reset(seed=0)
    frame = env.render_frame(env_idx=0, cell_size=16)
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8
    # The frame must contain non-background pixels (i.e. the maze is drawn).
    assert frame.max() > 32


@pytest.mark.skipif(
    not __import__("pathlib").Path.home().joinpath(".cache/mnist-maze/train-images-idx3-ubyte.gz").exists(),
    reason="real MNIST not cached locally",
)
def test_real_mnist_loader_smoke() -> None:
    images_by_class = load_mnist_by_class()
    assert len(images_by_class) == 10
    for bank in images_by_class:
        assert bank.dtype == np.uint8
        assert bank.shape[1:] == (28, 28)
