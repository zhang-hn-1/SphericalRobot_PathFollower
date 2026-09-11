"""Apply the small registration/import edits after new files are uploaded."""

from pathlib import Path


ROOT = Path("/data/lzq/workspace/SphericalRobot_PathFollower_20260911")


def insert_once(path, marker, addition):
    text = path.read_text()
    if addition.strip() in text:
        return
    if marker not in text:
        raise RuntimeError(f"Marker not found in {path}: {marker!r}")
    path.write_text(text.replace(marker, marker + addition, 1))


runner = ROOT / "legged_gym/dwl/on_policy_runner_dwl.py"
insert_once(
    runner,
    "from legged_gym.dwl.actor_critic_dwl import ActorCriticDWL\n",
    "from legged_gym.dwl.actor_critic_path_dwl import ActorCriticPathDWL\n",
)

vel_env = ROOT / "legged_gym/envs/rotunbot/vel_tracking/rotunbot_vel.py"
insert_once(
    vel_env,
    "            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))\n",
    "            if getattr(self, \"contact_yaw_damping_enabled\", False):\n"
    "                self.gym.refresh_rigid_body_state_tensor(self.sim)\n"
    "                self.gym.refresh_net_contact_force_tensor(self.sim)\n"
    "            self._apply_contact_yaw_damping()\n",
)
env_init = ROOT / "legged_gym/envs/__init__.py"
insert_once(
    env_init,
    "from .rotunbot.maze.rotunbot_maze_config import RotunbotMazeCfg, RotunbotMazeCfgPPO\n",
    "from .rotunbot.path_tracking.rotunbot_path import RotunbotPath\n"
    "from .rotunbot.path_tracking.rotunbot_path_config import RotunbotPathCfg, RotunbotPathCfgPPO\n",
)
insert_once(
    env_init,
    'task_registry.register( "rotunbot_maze", RotunbotMaze, RotunbotMazeCfg(), RotunbotMazeCfgPPO() )\n',
    'task_registry.register( "rotunbot_path", RotunbotPath, RotunbotPathCfg(), RotunbotPathCfgPPO() )\n',
)

print("Path follower imports and task registration installed.")


