"""Drive one Rotunbot through the procedural maze with WASD."""

import isaacgym  # noqa: F401 - Isaac Gym must be imported before torch
from isaacgym import gymapi
import torch

from legged_gym.envs import *  # noqa: F401,F403 - registers all tasks
from legged_gym.teleop import RotunbotKeyboardController
from legged_gym.utils import get_args, task_registry


KEY_BINDINGS = (
    (gymapi.KEY_W, "teleop_forward"),
    (gymapi.KEY_S, "teleop_reverse"),
    (gymapi.KEY_A, "teleop_left"),
    (gymapi.KEY_D, "teleop_right"),
    (gymapi.KEY_SPACE, "teleop_stop"),
    (gymapi.KEY_R, "teleop_reset"),
)


def teleop(args):
    if args.headless:
        raise ValueError("teleop_maze.py requires a viewer; remove --headless")

    args.task = "rotunbot_maze"
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 3600.0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.stop_distance = 0.0
    env_cfg.rewards.scales.to_target = 0.0

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if env.viewer is None:
        raise RuntimeError("Isaac Gym viewer was not created")

    controller = RotunbotKeyboardController(
        forward_speed=env_cfg.teleop.forward_speed,
        steering_position=env_cfg.teleop.steering_position,
    )
    for key, action_name in KEY_BINDINGS:
        env.gym.subscribe_viewer_keyboard_event(env.viewer, key, action_name)

    def handle_viewer_event(event):
        if controller.handle_event(event.action, event.value):
            print(controller.status())

    env.add_viewer_action_handler(handle_viewer_event)
    print("W/S: forward/reverse | A/D: left/right | Space: stop | R: reset | Esc: quit")
    print(controller.status())

    step_count = 0
    all_env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    while not env.gym.query_viewer_has_closed(env.viewer):
        if controller.consume_reset_request():
            env.reset_idx(all_env_ids)

        command = controller.normalized_action(
            env_cfg.control.first_actionScale,
            env_cfg.control.second_actionScale,
        )
        actions = torch.as_tensor(
            command, dtype=torch.float32, device=env.device
        ).unsqueeze(0)
        env.step(actions)
        step_count += 1

        if step_count % int(env_cfg.teleop.status_interval_steps) == 0:
            position = env.root_states[0, :3].detach().cpu().numpy()
            speed = torch.linalg.vector_norm(env.base_lin_vel[0]).item()
            print(
                f"position=({position[0]:+.2f}, {position[1]:+.2f}, "
                f"{position[2]:+.2f}) speed={speed:.2f} m/s"
            )


if __name__ == "__main__":
    teleop(get_args())
