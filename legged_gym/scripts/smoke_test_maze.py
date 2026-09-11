"""Create one maze and Rotunbot actor, then advance physics briefly."""

import os

import isaacgym  # noqa: F401 - Isaac Gym must be imported before torch
import torch

from legged_gym.envs import *  # noqa: F401,F403 - registers all tasks
from legged_gym.maps import wall_cells
from legged_gym.utils import get_args, task_registry


DEFAULT_STEPS = 300


def smoke_test(args):
    args.task = "rotunbot_maze"
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    actions = torch.zeros(
        (env.num_envs, env.num_actions), dtype=torch.float32, device=env.device
    )
    actions[:, 0] = 0.2

    steps = int(os.environ.get("ROTUNBOT_MAZE_SMOKE_STEPS", DEFAULT_STEPS))
    if steps <= 0:
        raise ValueError("ROTUNBOT_MAZE_SMOKE_STEPS must be positive")

    for _ in range(steps):
        env.step(actions)

    if not torch.isfinite(env.root_states).all():
        raise RuntimeError("non-finite Rotunbot state detected during maze smoke test")

    wall_count = len(wall_cells(env.maze_layout))
    position = env.root_states[0, :3].detach().cpu().numpy()
    print(
        "Maze smoke test passed: "
        f"steps={steps}, walls={wall_count}, "
        f"ball_position=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})"
    )


if __name__ == "__main__":
    smoke_test(get_args())
