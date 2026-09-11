"""Independent 50 Hz direct-joint RL task with held 5 Hz raw metric depth.

The original point-to-point class and its R torque law are not modified. World
translations used to isolate camera scenes are removed from map-coordinate
observations. No reference trajectory, velocity controller, or height encoder
is part of the actor's input.
"""
import copy
import math
import hashlib
import json
from types import SimpleNamespace
import os
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

from isaacgym import gymapi, gymtorch
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.rotunbot.target_point.rotunbot_target_repro import RotunbotTargetRepro
from ..motor_sru.camera_pose import mounted_camera_world_poses
from ..motor_sru.task_catalog import sha256, geometry_hash


def inverse_rotate(q, v):
    xyz, w = q[:, :3], q[:, 3:4]
    uv = torch.cross(xyz, v, dim=-1)
    return v - 2.0 * (w * uv - torch.cross(xyz, uv, dim=-1))


def wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def geometry_state(local_xy, centers, radius=.4, pillar_radius=.4, half_extent=8., bounds=None):
    clearance = torch.linalg.norm(centers - local_xy[:, None, :], dim=-1) - radius - pillar_radius
    if bounds is None:
        bounds = local_xy.new_tensor([-half_extent, half_extent, -half_extent, half_extent]).expand(len(local_xy), -1)
    boundaries = torch.stack((local_xy[:, 0] - bounds[:, 0] - radius,
                              bounds[:, 1] - local_xy[:, 0] - radius,
                              local_xy[:, 1] - bounds[:, 2] - radius,
                              bounds[:, 3] - local_xy[:, 1] - radius), dim=-1)
    return clearance, boundaries


def configuration_dict(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [configuration_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: configuration_dict(v) for k, v in obj.items()}
    return {k: configuration_dict(getattr(obj, k)) for k in dir(obj)
            if not k.startswith('_') and not callable(getattr(obj, k))}


def validate_pool(project_root, relative_root):
    """Read only the independent 40-pillar catalogs; never relax the old pool."""
    root = (Path(project_root) / relative_root).resolve()
    generation_path = root / 'generation.json'
    generation = json.loads(generation_path.read_text(encoding='utf-8'))
    catalogs, identities, seen_ids, seen_pairs = {}, {}, set(), {}
    for role in ('train', 'val', 'test', 'diagnostic'):
        path = root / 'data' / (role + '.json')
        declaration = generation['roles'][role]
        if (root / declaration['path']).resolve() != path.resolve():
            raise ValueError('Unexpected task manifest path for ' + role)
        digest = sha256(path)
        if digest != declaration['sha256']:
            raise ValueError('Task manifest checksum mismatch for ' + role)
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest['schema'] != 'depth_avoidance_maps_v1' or manifest['role'] != role:
            raise ValueError('Wrong 40-pillar task schema or split')
        tasks = manifest['tasks']
        if not tasks or len(tasks) != int(declaration['count']):
            raise ValueError('Task manifest count mismatch')
        stage_counts = {str(stage): 0 for stage in range(3)}
        for task in tasks:
            posts = np.asarray(task['posts'], dtype=np.float64)
            start, goal = np.asarray(task['start'], dtype=np.float64), np.asarray(task['goal'], dtype=np.float64)
            bounds = np.asarray(task['map_bounds'], dtype=np.float64)
            if posts.shape != (40, 2) or start.shape != (2,) or goal.shape != (2,) or bounds.shape != (4,):
                raise ValueError('Forty finite pillar centers and 2D endpoints are required')
            if not np.isfinite(np.concatenate((posts.ravel(), start, goal, bounds, [task['yaw']]))).all():
                raise ValueError('Nonfinite task geometry')
            if int(task['stage']) != task['stage'] or task['stage'] not in (0, 1, 2):
                raise ValueError('Invalid explicit task stage')
            if task['category'] not in ('required_detour', 'clear_direct'):
                raise ValueError('Unexpected task category')
            if task.get('split', role) != role:
                raise ValueError('Task split disagrees with its manifest')
            if any(abs(float(task[key]) - value) > 1e-9 for key, value in
                   (('robot_radius_m', .4), ('post_radius_m', .4), ('post_height_m', 1.5))):
                raise ValueError('Robot and pillar dimensions differ from this simulator')
            if task['geometry_sha256'] != geometry_hash(task):
                raise ValueError('Task geometry checksum mismatch')
            if np.linalg.norm(start - goal) <= .4:
                raise ValueError('Task starts within the success region')
            for endpoint in (start, goal):
                if np.linalg.norm(posts - endpoint, axis=1).min() <= .8:
                    raise ValueError('Task endpoint overlaps a pillar')
                if min(endpoint[0]-bounds[0], bounds[1]-endpoint[0], endpoint[1]-bounds[2], bounds[3]-endpoint[1]) <= .4:
                    raise ValueError('Task endpoint overlaps a map boundary')
            if task['task_id'] in seen_ids:
                raise ValueError('Duplicate task id across splits')
            seen_ids.add(task['task_id'])
            pair_id = task['pair_id']
            if pair_id in seen_pairs and seen_pairs[pair_id] != role:
                raise ValueError('Paired layouts leak across train and evaluation splits')
            seen_pairs[pair_id] = role
            stage_counts[str(int(task['stage']))] += 1
        catalogs[role] = SimpleNamespace(tasks=tasks, path=path.resolve(), manifest_sha256=digest)
        identities[role] = dict(sha256=digest, count=len(tasks), stage_counts=stage_counts)
    return catalogs, dict(schema='depth_avoidance_maps_v1', generation_sha256=sha256(generation_path), roles=identities)


class StageTaskSampler:
    """No-replacement cycles through a fixed, checkpointed stage filter."""
    def __init__(self, tasks, seed, max_stage=0, ordered_first_batch=False):
        self.max_stage = max_stage
        self.eligible = np.asarray([i for i, task in enumerate(tasks)
                                    if max_stage is None or int(task['stage']) <= max_stage], dtype=np.int64)
        if not len(self.eligible):
            raise ValueError('The selected maximum stage contains no tasks')
        self.rng = np.random.RandomState(int(seed))
        self.order = self.eligible.copy() if ordered_first_batch else self.rng.permutation(self.eligible)
        self.cursor = 0

    def draw(self, count):
        chunks, remaining = [], int(count)
        if remaining < 0:
            raise ValueError('Negative task count')
        while remaining:
            if self.cursor == len(self.order):
                self.order, self.cursor = self.rng.permutation(self.eligible), 0
            take = min(remaining, len(self.order) - self.cursor)
            chunks.append(self.order[self.cursor:self.cursor+take])
            self.cursor += take
            remaining -= take
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

    def state_dict(self):
        rng = self.rng.get_state()
        return dict(max_stage=self.max_stage, eligible=self.eligible.tolist(), order=self.order.tolist(),
                    cursor=int(self.cursor), rng=[rng[0], rng[1].tolist(), int(rng[2]), int(rng[3]), float(rng[4])])

    def load_state_dict(self, state):
        if state['max_stage'] != self.max_stage or state['eligible'] != self.eligible.tolist():
            raise ValueError('Sampler stage filter changed; start a new experiment instead of resuming')
        order = np.asarray(state['order'], dtype=np.int64)
        if sorted(order.tolist()) != sorted(self.eligible.tolist()) or not 0 <= int(state['cursor']) <= len(order):
            raise ValueError('Invalid saved task-sampler cycle')
        rng = state['rng']
        self.rng.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), int(rng[2]), int(rng[3]), float(rng[4])))
        self.order, self.cursor = order, int(state['cursor'])


def reward_terms(previous_distance, distance, success, collision, out_of_bounds, unstable,
                 tilt, speed, torques, targets, previous_targets, dt, rewards, stop_speed=.1):
    """Explicit units; progress telescopes and stopping has no alive bonus."""
    scales = rewards.scales
    bounds_only = out_of_bounds & ~collision
    unstable_only = unstable & ~collision & ~out_of_bounds
    tilt_excess = (tilt.abs() - float(rewards.excessive_tilt_start_rad)).clamp_min(0.)
    speed_excess = (speed - stop_speed).clamp_min(0.)
    terms = dict(
        progress_per_m=float(scales.progress_per_m) * (previous_distance - distance),
        success_once=float(scales.success_once) * success.float(),
        collision_once=float(scales.collision_once) * collision.float(),
        bounds_once=float(scales.bounds_once) * bounds_only.float(),
        unstable_once=float(scales.unstable_once) * unstable_only.float(),
        time_per_s=torch.full_like(distance, float(scales.time_per_s) * dt),
        excessive_tilt_per_s=float(scales.excessive_tilt_per_s) * tilt_excess.square().sum(-1) * dt,
        near_goal_speed_per_s=float(scales.near_goal_speed_per_s) * (distance <= rewards.near_goal_region_m).float() * speed_excess.square() * dt,
        torques_squared_per_s=float(scales.torques_squared_per_s) * torques.square().sum(-1) * dt,
        target_change_squared_per_step=float(scales.target_change_squared_per_step) * (targets-previous_targets).square().sum(-1))
    total = sum(terms.values())
    if not torch.isfinite(total).all():
        raise RuntimeError('Nonfinite depth RL reward')
    return total, terms


class MotorDepthEnv(RotunbotTargetRepro):
    category_names = ('required_detour', 'clear_direct')

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless=True):
        if sim_device not in ('cuda:2', 'cuda:3') or 'CUDA_VISIBLE_DEVICES' in os.environ:
            raise ValueError('Motor task requires unremapped physical cuda:2 or cuda:3')
        if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
            raise ValueError('CUDA_DEVICE_ORDER must be PCI_BUS_ID before CUDA initialization')
        if not sim_params.use_gpu_pipeline:
            raise ValueError('Real depth requires the GPU pipeline')
        if abs(float(sim_params.dt) - .02) > 1e-6 or cfg.control.decimation != 1 or cfg.control.control_type != 'R':
            raise ValueError('Preserve the 50 Hz R executor/physics baseline')
        if cfg.env.num_observations != 380 or cfg.env.num_privileged_obs != 63:
            raise ValueError('Parent proprioception buffers must remain 380/63')
        if cfg.navigation.pillar_count != 40 or (cfg.camera.height, cfg.camera.width) != (40, 64):
            raise ValueError('The new profile requires exactly 40 pillars and 40x64 depth')
        if cfg.camera.motor_steps_per_frame != 10:
            raise ValueError('The camera must remain at 5 Hz with a 50 Hz motor policy')
        if abs(cfg.control.first_actionScale - 3.) > 1e-9 or abs(cfg.control.second_actionScale - .45) > 1e-9:
            raise ValueError('Normalized actions require one R-executor scale of [3,.45]')
        if abs(cfg.env.episode_length_s - 180.) > 1e-8 or cfg.commands.target_curriculum:
            raise ValueError('Motor pillar admission requires 180 seconds and no implicit curriculum')
        # The legacy unused PID allocates on bare cuda; select an authorized GPU
        # before calling its constructor instead of changing the original class.
        torch.cuda.set_device(torch.device(sim_device))
        self.project_root = Path(LEGGED_GYM_ROOT_DIR)
        self.catalogs, self.pool_identity = validate_pool(
            self.project_root, cfg.task_catalog.pool_root)
        self.catalog = self.catalogs[cfg.task_catalog.role]
        if (self.project_root / cfg.task_catalog.manifest_path).resolve() != self.catalog.path:
            raise ValueError('Active manifest must be the validated split from this pool')
        self.sampler = StageTaskSampler(self.catalog.tasks, cfg.task_catalog.sample_seed,
                                        cfg.task_catalog.max_stage, cfg.task_catalog.ordered_first_batch)
        fixed = cfg.task_catalog.fixed_task_indices
        if fixed is None:
            self._task_indices_cpu = self.sampler.draw(cfg.env.num_envs)
        else:
            if len(fixed) != cfg.env.num_envs or any(int(i) != i or not 0 <= i < len(self.catalog.tasks) for i in fixed):
                raise ValueError('One valid fixed task index is required per environment')
            if any(int(i) not in self.sampler.eligible for i in fixed):
                raise ValueError('Fixed task indices must satisfy the active stage filter')
            self._task_indices_cpu = np.asarray(fixed, dtype=np.int64)
        self._constructing = True
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.max_episode_length = 9000
        self.max_episode_length_s = 180.
        self.training_success_distance = .4
        self.num_obs, self.num_privileged_obs = 2943, 188
        self.data_print = False
        self._camera_depth_tensors = [gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(
            self.sim, env, camera, gymapi.IMAGE_DEPTH)) for env, camera in zip(self.envs, self._camera_handles)]
        self._constructing = False
        self._reset_to_tasks(torch.arange(self.num_envs, device=self.device), resample=False)
        # Initialize articulation renderer transforms outside the task clock.
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._reset_to_tasks(torch.arange(self.num_envs, device=self.device), resample=False)
        self.compute_observations()
        self._training_contract = self._build_training_contract()

    def _prepare_reward_function(self):
        # Avoid the inherited automatic dt conversion and inherited shaping.
        self.reward_scales = configuration_dict(self.cfg.rewards.scales)
        self.reward_functions = []
        self.reward_names = list(self.reward_scales)
        self.episode_sums = {name: torch.zeros(self.num_envs, device=self.device) for name in self.reward_names}

    def _normalized_motor_targets(self, targets=None):
        if targets is None:
            targets = self.output_actions
        return targets / targets.new_tensor([3., .45])

    def _legacy_raw_actions(self):
        # Preserve the original 19-D body-history action units despite the new
        # normalized action distribution. Original joint2 scale was 0.5.
        return self.actions * self.actions.new_tensor([3., .9])

    def create_sim(self):
        # BaseTask disables graphics when headless; cameras still need graphics.
        self.graphics_device_id = self.sim_device_id
        return super().create_sim()

    def _get_env_origins(self):
        if self.cfg.terrain.mesh_type != 'plane' or self.cfg.env.env_spacing < 30.:
            raise ValueError('A plane with >=30 m scene isolation is required')
        self.custom_origins = False
        columns = max(int(math.sqrt(self.num_envs)), 1)
        indices = torch.arange(self.num_envs, device=self.device)
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self.env_origins[:, 0] = torch.div(indices, columns, rounding_mode='floor') * self.cfg.env.env_spacing
        self.env_origins[:, 1] = (indices % columns) * self.cfg.env.env_spacing
        # PhysX's plane is infinite, but its graphical ground has finite extent.
        # Center the occupied grid without changing spacing or any local task.
        midpoint = (self.env_origins[:, :2].amin(dim=0) + self.env_origins[:, :2].amax(dim=0)) / 2.
        self.env_origins[:, :2] -= midpoint

    def _create_envs(self):
        path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        options = gymapi.AssetOptions()
        for key in ('default_dof_drive_mode', 'collapse_fixed_joints', 'replace_cylinder_with_capsule',
                    'flip_visual_attachments', 'fix_base_link', 'density', 'angular_damping', 'linear_damping',
                    'max_angular_velocity', 'max_linear_velocity', 'armature', 'thickness', 'disable_gravity'):
            setattr(options, key, getattr(self.cfg.asset, key))
        robot = self.gym.load_asset(self.sim, str(Path(path).parent), Path(path).name, options)
        self.num_dof = self.gym.get_asset_dof_count(robot)
        names = self.gym.get_asset_rigid_body_names(robot)
        self.num_bodies = len(names)
        self.dof_names = self.gym.get_asset_dof_names(robot)
        self.num_dofs = len(self.dof_names)
        if self.num_dof != 2 or names[0] != 'base_link':
            raise ValueError('Expected original two-joint robot rooted at base_link')
        dof_props = self.gym.get_asset_dof_properties(robot)
        shapes = self.gym.get_asset_rigid_shape_properties(robot)
        state = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = torch.tensor(state, device=self.device, dtype=torch.float32)
        self._get_env_origins()
        pillar_options = gymapi.AssetOptions()
        pillar_options.fix_base_link = True
        pillar_options.replace_cylinder_with_capsule = False
        pillar = self.gym.load_asset(self.sim, str(Path(__file__).parent.parent / 'motor_sru'), 'pillar.urdf', pillar_options)
        if self.gym.get_asset_dof_count(pillar) != 0 or self.gym.get_asset_rigid_body_count(pillar) != 1:
            raise ValueError('Pillars must be single fixed bodies without DOFs')
        camera_props = gymapi.CameraProperties()
        camera_props.width, camera_props.height = self.cfg.camera.width, self.cfg.camera.height
        camera_props.horizontal_fov = self.cfg.camera.horizontal_fov
        camera_props.near_plane, camera_props.far_plane = self.cfg.camera.near_plane, self.cfg.camera.far_plane
        camera_props.enable_tensors = True
        self.envs, self.actor_handles, self._camera_handles = [], [], []
        self._robot_actor_indices_cpu, self._pillar_actor_indices_cpu = [], []
        self.env_frictions = torch.zeros(self.num_envs, 1, device=self.device)
        origins = self.env_origins.cpu().numpy()
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, gymapi.Vec3(0., 0., 0.), gymapi.Vec3(0., 0., 0.), max(int(math.sqrt(self.num_envs)), 1))
            task = self.catalog.tasks[self._task_indices_cpu[i]]
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(float(origins[i, 0] + task['start'][0]), float(origins[i, 1] + task['start'][1]), float(self.cfg.init_state.pos[2]))
            pose.r = gymapi.Quat(0., 0., math.sin(task['yaw'] / 2), math.cos(task['yaw'] / 2))
            self.gym.set_asset_rigid_shape_properties(robot, self._process_rigid_shape_props(shapes, i))
            actor = self.gym.create_actor(env, robot, pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            self.gym.set_actor_dof_properties(env, actor, self._process_dof_props(dof_props, i))
            body_props = self._process_rigid_body_props(self.gym.get_actor_rigid_body_properties(env, actor), i)
            self.gym.set_actor_rigid_body_properties(env, actor, body_props, recomputeInertia=True)
            self.envs.append(env)
            self.actor_handles.append(actor)
            self._robot_actor_indices_cpu.append(self.gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM))
            pillar_indices = []
            for j, (x, y) in enumerate(task['posts']):
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(float(origins[i, 0] + x), float(origins[i, 1] + y), .75)
                handle = self.gym.create_actor(env, pillar, pose, 'pillar_{}'.format(j), i, 0, 0)
                pillar_indices.append(self.gym.get_actor_index(env, handle, gymapi.DOMAIN_SIM))
            self._pillar_actor_indices_cpu.append(pillar_indices)
            self._camera_handles.append(self.gym.create_camera_sensor(env, camera_props))
        actors_per_env = 1 + self.cfg.navigation.pillar_count
        if self._robot_actor_indices_cpu != list(range(0, self.num_envs * actors_per_env, actors_per_env)):
            raise RuntimeError('Robot/pillar ordering differs from the explicit 41-actor contract')
        for attribute, filters in (('feet_indices', [self.cfg.asset.foot_name]),
                                   ('penalised_contact_indices', self.cfg.asset.penalize_contacts_on),
                                   ('termination_contact_indices', self.cfg.asset.terminate_after_contacts_on)):
            indices = [j for j, name in enumerate(names) if any(fragment in name for fragment in filters)]
            setattr(self, attribute, torch.tensor(indices, dtype=torch.long, device=self.device))
        self._gym_camera_env_origins = np.asarray([[o.x, o.y, o.z] for o in (self.gym.get_env_origin(e) for e in self.envs)])

    def _init_buffers(self):
        # Parent assumes one root actor per env. Allocate the same robot buffers
        # against the explicit robot view; never pass all 41 actors to its init.
        self._all_root_states = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.dof_state = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self._all_contact_forces = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim)).view(self.num_envs, self.num_bodies + self.cfg.navigation.pillar_count, 3)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.root_states = self._all_root_states.view(self.num_envs, 1 + self.cfg.navigation.pillar_count, 13)[:, 0, :]
        self.base_pos, self.base_quat = self.root_states[:, :3], self.root_states[:, 3:7]
        self.contact_forces = self._all_contact_forces[:, :self.num_bodies, :]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.robot_actor_indices = torch.tensor(self._robot_actor_indices_cpu, dtype=torch.int32, device=self.device)
        self.pillar_actor_indices = torch.tensor(self._pillar_actor_indices_cpu, dtype=torch.long, device=self.device)
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = torch.tensor([0., 0., -1.], device=self.device).repeat(self.num_envs, 1)
        self.forward_vec = torch.tensor([1., 0., 0.], device=self.device).repeat(self.num_envs, 1)
        for name in ('torques', 'actions', 'last_actions', 'output_actions', 'last_output_actions', 'last_dof_vel', '_requested_raw_actions'):
            setattr(self, name, torch.zeros(self.num_envs, 2, device=self.device))
        self.p_gains, self.d_gains = torch.zeros(2, device=self.device), torch.zeros(2, device=self.device)
        self.last_root_vel = torch.zeros(self.num_envs, 6, device=self.device)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, device=self.device)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device)
        self.feet_air_time = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device)
        self.last_contacts = torch.zeros_like(self.feet_air_time, dtype=torch.bool)
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel = torch.zeros_like(self.base_lin_vel)
        self.projected_gravity = self.gravity_vec.clone()
        self.measured_heights = 0
        self.default_dof_pos = torch.tensor([[self.cfg.init_state.default_joint_angles[n] for n in self.dof_names]], device=self.device)
        for name in ('goal_dist', 'last_goal_dist', 'orientation_error', 'last_orientation_error', 'terminal_goal_dist', 'terminal_speed', 'terminal_balance_reward', 'episode_returns', 'actual_yaw_rate', 'previous_yaw'):
            setattr(self, name, torch.zeros(self.num_envs, device=self.device))
        for name in ('success_buf', 'arrived_target_buf', 'stop_buf', 'collision_buf', 'physical_pillar_contact', 'bounds_buf', 'unstable_buf', 'terminal_timeout', 'terminal_unstable', 'terminal_out_of_bounds', 'depth_fresh'):
            setattr(self, name, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        self.terminal_position = torch.zeros(self.num_envs, 2, device=self.device)
        self.base_euler_tensor = torch.zeros(self.num_envs, 3, device=self.device)
        self.obs_history = deque([torch.zeros(self.num_envs, 19, device=self.device) for _ in range(20)], maxlen=20)
        self.critic_history = deque([torch.zeros(self.num_envs, 21, device=self.device) for _ in range(3)], maxlen=3)
        self.task_indices = torch.as_tensor(self._task_indices_cpu.copy(), device=self.device)
        self.task_ids = [self.catalog.tasks[i]['task_id'] for i in self._task_indices_cpu]
        self.task_starts = torch.zeros(self.num_envs, 2, device=self.device)
        self.task_local_goals = torch.zeros_like(self.task_starts)
        self.pillar_centers = torch.zeros(self.num_envs, self.cfg.navigation.pillar_count, 2, device=self.device)
        self.task_bounds = torch.zeros(self.num_envs, 4, device=self.device)
        self.task_stages = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.task_reset_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.category_id = torch.zeros_like(self.task_reset_count)
        self.depth_image = torch.zeros(self.num_envs, 40, 64, device=self.device)
        self.depth_stats = torch.zeros(self.num_envs, 3, device=self.device)
        self.capture_camera_pose = torch.zeros(self.num_envs, 7, device=self.device)
        self.depth_capture_count = torch.zeros_like(self.task_reset_count)
        self.depth_age = torch.full_like(self.task_reset_count, 10)
        self._proprio_pending = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._depth_pending = self._proprio_pending.clone()
        self.training_success_distance = self.cfg.evaluation.target_error_threshold
        self.target_curriculum_successes = self.target_curriculum_attempts = 0
        self.target_curriculum_last_success_rate = 0.

    def _sync_robot_state(self, ids=None):
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        self.base_lin_vel[ids] = inverse_rotate(self.base_quat[ids], self.root_states[ids, 7:10])
        self.base_ang_vel[ids] = inverse_rotate(self.base_quat[ids], self.root_states[ids, 10:13])
        self.projected_gravity[ids] = inverse_rotate(self.base_quat[ids], self.gravity_vec[ids])
        self._update_base_euler()

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.
        indices = self.robot_actor_indices[env_ids].contiguous()
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(indices), len(indices))

    def _resample_commands(self, env_ids):
        self.commands[env_ids] = 0.
        self.commands[env_ids, :2] = self.task_local_goals[env_ids] + self.env_origins[env_ids, :2]

    def _reset_to_tasks(self, env_ids, resample=True):
        if env_ids.numel() == 0:
            return
        cpu_ids = env_ids.detach().cpu().numpy()
        if resample and self.cfg.task_catalog.resample_on_reset:
            self._task_indices_cpu[cpu_ids] = self.sampler.draw(len(cpu_ids))
        tasks = [self.catalog.tasks[self._task_indices_cpu[i]] for i in cpu_ids]
        tensor = lambda key: torch.tensor([t[key] for t in tasks], dtype=torch.float32, device=self.device)
        self.task_indices[env_ids] = torch.tensor(self._task_indices_cpu[cpu_ids], device=self.device)
        for i, task in zip(cpu_ids, tasks):
            self.task_ids[int(i)] = task['task_id']
        self.category_id[env_ids] = torch.tensor([int(t['category'] == 'clear_direct') for t in tasks], device=self.device)
        self.task_starts[env_ids], self.task_local_goals[env_ids] = tensor('start'), tensor('goal')
        self.pillar_centers[env_ids] = tensor('posts')
        self.task_bounds[env_ids] = tensor('map_bounds')
        self.task_stages[env_ids] = torch.tensor([int(t['stage']) for t in tasks], device=self.device)
        pillar_ids = self.pillar_actor_indices[env_ids].flatten()
        states = torch.zeros(len(pillar_ids), 13, device=self.device)
        states[:, :2] = (self.pillar_centers[env_ids] + self.env_origins[env_ids, None, :2]).reshape(-1, 2)
        states[:, 2], states[:, 6] = .75, 1.
        self._all_root_states[pillar_ids] = states
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, :2] = self.task_starts[env_ids] + self.env_origins[env_ids, :2]
        self.root_states[env_ids, 3:7] = 0.
        yaw = tensor('yaw')
        self.root_states[env_ids, 5], self.root_states[env_ids, 6] = torch.sin(yaw / 2), torch.cos(yaw / 2)
        self.root_states[env_ids, 7:13] = 0.
        indices = torch.cat((self.robot_actor_indices[env_ids].long(), pillar_ids)).to(torch.int32).contiguous()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._all_root_states), gymtorch.unwrap_tensor(indices), len(indices))
        self._reset_dofs(env_ids)
        self._resample_commands(env_ids)
        for name in ('actions', 'last_actions', 'output_actions', 'last_output_actions', 'torques', 'last_dof_vel', 'last_root_vel', '_requested_raw_actions',
                     'last_base_lin_vel', 'last_base_ang_vel', 'last_root_states', 'orientation_error', 'last_orientation_error',
                     'feet_air_time', 'last_contacts', 'episode_length_buf', 'episode_returns', 'success_buf', 'arrived_target_buf',
                     'stop_buf', 'collision_buf', 'physical_pillar_contact', 'bounds_buf', 'unstable_buf', 'actual_yaw_rate',
                     'terminal_goal_dist', 'terminal_speed', 'terminal_balance_reward', 'terminal_position', 'terminal_timeout',
                     'terminal_unstable', 'terminal_out_of_bounds', 'depth_image', 'depth_stats', 'capture_camera_pose'):
            if hasattr(self, name):
                getattr(self, name)[env_ids] = 0
        for history in tuple(self.obs_history) + tuple(self.critic_history):
            history[env_ids] = 0.
        for value in self.episode_sums.values():
            value[env_ids] = 0.
        self._all_contact_forces[env_ids] = 0.
        for name in ('PID_FirstAxis', 'PID_SecondAxis'):
            if hasattr(self, name):
                getattr(self, name).reset(env_ids)
        self._sync_robot_state(env_ids)
        self.previous_yaw[env_ids] = self.base_euler_tensor[env_ids, 2]
        self.goal_dist[env_ids] = torch.linalg.norm(self.commands[env_ids, :2] - self.root_states[env_ids, :2], dim=-1)
        self.last_goal_dist[env_ids] = self.goal_dist[env_ids]
        self.reset_buf[env_ids], self.time_out_buf[env_ids] = 0, False
        self.depth_fresh[env_ids] = False
        self.depth_age[env_ids] = 10
        self._depth_pending[env_ids], self._proprio_pending[env_ids] = True, True
        self.task_reset_count[env_ids] += 1

    @torch.inference_mode()
    def reset_idx(self, env_ids):
        self._reset_to_tasks(env_ids, resample=True)

    @torch.inference_mode()
    def reset(self):
        self._reset_to_tasks(torch.arange(self.num_envs, device=self.device), resample=False)
        self.rew_buf.zero_()
        self.compute_observations()
        return self.get_observations(), self.get_privileged_observations()

    @torch.inference_mode()
    def _capture_depth(self, ids):
        poses = mounted_camera_world_poses(self.root_states.detach().cpu().numpy(), self.cfg.camera.position, self.cfg.camera.rotation)
        for env, camera, pose, gym_origin in zip(self.envs, self._camera_handles, poses, self._gym_camera_env_origins):
            transform = gymapi.Transform()
            transform.p = gymapi.Vec3(*[float(x) for x in pose[:3] - gym_origin])
            transform.r = gymapi.Quat(*[float(x) for x in pose[3:]])
            self.gym.set_camera_transform(camera, env, transform)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        try:
            raw = torch.stack([self._camera_depth_tensors[int(i)] for i in ids.cpu().tolist()]).clone()
        finally:
            self.gym.end_access_image_tensors(self.sim)
        metric = -raw
        valid = torch.isfinite(metric) & (metric >= self.cfg.camera.near_plane) & (metric <= self.cfg.camera.far_plane)
        metric = torch.where(valid, metric, torch.zeros_like(metric))
        if self.cfg.camera.horizontal_flip_to_robot_frame:
            metric = metric.flip(-1)
        if metric.shape != (len(ids), 40, 64) or not torch.isfinite(metric).all():
            raise RuntimeError('Invalid raw metric depth shape/values')
        self.depth_image[ids] = metric
        self.depth_stats[ids] = torch.stack((valid.float().mean((1, 2)), metric.mean((1, 2)), metric.amax((1, 2))), dim=-1)
        self.capture_camera_pose[ids] = torch.tensor(poses, dtype=torch.float32, device=self.device)[ids]
        self.depth_fresh[ids], self._depth_pending[ids] = True, False
        self.depth_age[ids] = 0
        self.depth_capture_count[ids] += 1

    @torch.inference_mode()
    def compute_observations(self):
        ids = self._proprio_pending.nonzero(as_tuple=False).flatten()
        if len(ids):
            self._sync_robot_state(ids)
            local_position = self.root_states[ids, :3] - self.env_origins[ids]
            goal = self.commands[ids, :2] - self.env_origins[ids, :2]
            s = self.obs_scales
            actor = torch.cat((goal * s.command, local_position[:, :2] * s.pos, self.base_quat[ids] * s.quat,
                               self.base_lin_vel[ids] * s.lin_vel, self.base_ang_vel[ids] * s.ang_vel,
                               self.dof_pos[ids, 1:2] * s.dof_pos, self.dof_vel[ids] * s.dof_vel, self._legacy_raw_actions()[ids]), dim=-1)
            critic = torch.cat((goal * s.command, local_position * s.pos, self.base_quat[ids] * s.quat,
                                self.base_lin_vel[ids] * s.lin_vel, self.base_ang_vel[ids] * s.ang_vel,
                                self.dof_pos[ids] * s.dof_pos, self.dof_vel[ids] * s.dof_vel, self._legacy_raw_actions()[ids]), dim=-1)
            if self.add_noise:
                actor = actor + (2. * torch.rand_like(actor) - 1.) * self.noise_scale_vec
            for history, current in ((self.obs_history, actor), (self.critic_history, critic)):
                for j in range(len(history) - 1):
                    history[j][ids] = history[j + 1][ids]
                history[-1][ids] = current
            self._proprio_pending[ids] = False
        self.obs_buf = torch.stack(list(self.obs_history), dim=1).reshape(self.num_envs, 380).clamp(-self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)
        self.privileged_obs_buf = torch.cat(list(self.critic_history), dim=-1).clamp(-self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)
        due = (self._depth_pending | (self.depth_age >= self.cfg.camera.motor_steps_per_frame)).nonzero(as_tuple=False).flatten()
        if len(due) and not self._constructing:
            self._capture_depth(due)

    @torch.inference_mode()
    def get_observations(self):
        self.compute_observations()
        return torch.cat((self.obs_buf, self.depth_image.flatten(1), self._normalized_motor_targets(),
                          self.depth_fresh[:, None].float()), dim=-1)

    @torch.inference_mode()
    def get_privileged_observations(self):
        self.compute_observations()
        xy = self.root_states[:, :2] - self.env_origins[:, :2]
        relative = self.pillar_centers - xy[:, None, :]
        yaw = self.base_euler_tensor[:, 2:3]
        c, s = torch.cos(yaw), torch.sin(yaw)
        relative = torch.stack((c * relative[:, :, 0] + s * relative[:, :, 1],
                                -s * relative[:, :, 0] + c * relative[:, :, 1]), dim=-1)
        radii = torch.full((self.num_envs, self.cfg.navigation.pillar_count, 1), self.cfg.navigation.pillar_radius, device=self.device)
        _, boundaries = geometry_state(xy, self.pillar_centers, self.cfg.navigation.robot_radius,
                                               self.cfg.navigation.pillar_radius, bounds=self.task_bounds)
        remaining_time = (1. - self.episode_length_buf.to(self.privileged_obs_buf.dtype)
                          / float(self.max_episode_length)).clamp(0., 1.).unsqueeze(-1)
        return torch.cat((self.privileged_obs_buf, torch.cat((relative, radii), dim=-1).flatten(1),
                          boundaries, remaining_time), dim=-1)

    def check_termination(self):
        xy = self.root_states[:, :2] - self.env_origins[:, :2]
        clearance, boundaries = geometry_state(xy, self.pillar_centers, self.cfg.navigation.robot_radius,
                                               self.cfg.navigation.pillar_radius, bounds=self.task_bounds)
        self.physical_pillar_contact[:] = (torch.linalg.norm(self._all_contact_forces[:, self.num_bodies:, :], dim=-1)
                                           > self.cfg.navigation.pillar_contact_force_threshold).any(dim=-1)
        self.collision_buf[:] = (clearance.amin(dim=-1) <= 0.) | self.physical_pillar_contact
        self.bounds_buf[:] = boundaries.amin(dim=-1) < 0.
        self.unstable_buf[:] = (self.base_euler_tensor[:, :2].abs() > self.cfg.navigation.instability_angle).any(dim=-1)
        self.time_out_buf[:] = self.episode_length_buf >= self.max_episode_length
        speed = torch.linalg.norm(self.root_states[:, 7:10], dim=-1)
        self.arrived_target_buf[:] = self.goal_dist <= self.cfg.evaluation.target_error_threshold
        self.stop_buf[:] = speed <= self.cfg.evaluation.stop_velocity_threshold
        self.success_buf[:] = self.arrived_target_buf & self.stop_buf & ~self.collision_buf & ~self.bounds_buf & ~self.unstable_buf
        self.reset_buf[:] = self.success_buf | self.collision_buf | self.bounds_buf | self.unstable_buf | self.time_out_buf
        self.actual_yaw_rate[:] = wrap_angle(self.base_euler_tensor[:, 2] - self.previous_yaw) / self.dt
        poses = mounted_camera_world_poses(self.root_states.detach().cpu().numpy(), self.cfg.camera.position, self.cfg.camera.rotation)
        self.extras['motor_navigation'] = dict(
            task_index=self.task_indices.clone(), task_ids=list(self.task_ids), category_id=self.category_id.clone(),
            task_stage=self.task_stages.clone(), stage_id=self.task_stages.clone(),
            success=self.success_buf.clone(), collision=self.collision_buf.clone(), out_of_bounds=self.bounds_buf.clone(),
            physical_pillar_contact=self.physical_pillar_contact.clone(),
            unstable=self.unstable_buf.clone(), timeout=self.time_out_buf.clone(), goal_distance=self.goal_dist.clone(),
            speed=speed.clone(), local_xy=xy.clone(), yaw=self.base_euler_tensor[:, 2].clone(),
            episode_length=self.episode_length_buf.clone(), raw_actions=self._requested_raw_actions.clone(),
            normalized_actions=self.actions.clone(), motor_targets=self.output_actions.clone(),
            previous_motor_targets=self.last_output_actions.clone(),
            normalized_motor_targets=self._normalized_motor_targets().clone(),
            roll=self.base_euler_tensor[:, 0].clone(), pitch=self.base_euler_tensor[:, 1].clone(),
            actual_v_world_xy=self.root_states[:, 7:9].clone(), actual_yaw_rate=self.actual_yaw_rate.clone(),
            radius_constraint_enabled=False,
            minimum_clearance=clearance.amin(dim=-1).clone(), depth_fresh=self.depth_fresh.clone(),
            depth_stats=self.depth_stats.clone(), capture_camera_pose=self.capture_camera_pose.clone(),
            camera_world_pose=torch.tensor(poses, dtype=torch.float32, device=self.device))

    def compute_reward(self):
        reward, terms = reward_terms(
            self.last_goal_dist, self.goal_dist, self.success_buf, self.collision_buf,
            self.bounds_buf, self.unstable_buf, self.base_euler_tensor[:, :2],
            torch.linalg.norm(self.root_states[:, 7:10], dim=-1), self.torques,
            self._normalized_motor_targets(), self._normalized_motor_targets(self.last_output_actions),
            self.dt, self.cfg.rewards, self.cfg.evaluation.stop_velocity_threshold)
        self.rew_buf[:] = reward
        for name, term in terms.items():
            self.episode_sums[name] += term
        self.extras['motor_navigation']['reward_terms'] = {name: term.clone() for name, term in terms.items()}
        self.extras['motor_navigation']['reward'] = reward.clone()

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._sync_robot_state()
        self._post_physics_step_callback()
        self.check_termination()
        self.compute_reward()
        self.episode_returns += self.rew_buf
        self.extras['motor_navigation']['episode_return'] = self.episode_returns.clone()
        done = self.reset_buf.clone()
        timeouts = self.time_out_buf.clone()
        reset_ids = done.nonzero(as_tuple=False).flatten()
        self.previous_yaw[:] = self.base_euler_tensor[:, 2]
        if len(reset_ids):
            self.extras['episode'] = {name: self.episode_sums[name][reset_ids].mean().item() for name in self.episode_sums}
            for value in self.episode_sums.values():
                value[reset_ids] = 0.
            self._reset_to_tasks(reset_ids, resample=True)
        self.reset_buf[:] = done
        self.extras['time_outs'] = timeouts
        self._proprio_pending[:] = True
        self.depth_age += 1
        self.depth_fresh[:] = False
        self.compute_observations()
        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 7:13]

    @torch.inference_mode()
    def step(self, actions):
        if actions.shape != (self.num_envs, 2) or not torch.isfinite(actions).all():
            raise ValueError('Motor actions must be finite [N,2] normalized joint commands')
        if (actions.abs() > 1.000001).any():
            raise ValueError('Normalized motor actions must stay within [-1,1]')
        actions = actions.to(self.device).clamp(-1., 1.)
        self._requested_raw_actions.copy_(actions.to(self.device))
        _, _, reward, done, info = super().step(actions)
        return self.get_observations(), self.get_privileged_observations(), reward.clone(), done.bool().clone(), info

    def export_task_state(self):
        return dict(tasks=[copy.deepcopy(self.catalog.tasks[int(i)]) for i in self._task_indices_cpu],
                    task_indices=self._task_indices_cpu.tolist(), runtime=self.get_runtime_metadata(),
                    camera=configuration_dict(self.cfg.camera), pool_identity=copy.deepcopy(self.pool_identity))

    def get_runtime_metadata(self):
        return dict(role=self.cfg.task_catalog.role, num_envs=self.num_envs,
                    sample_seed=self.cfg.task_catalog.sample_seed,
                    max_stage=self.cfg.task_catalog.max_stage,
                    eligible_task_indices=self.sampler.eligible.tolist(),
                    fixed_task_indices=self.cfg.task_catalog.fixed_task_indices,
                    resample_on_reset=self.cfg.task_catalog.resample_on_reset,
                    sampler=self.sampler.state_dict(),
                    manifest_sha256=self.catalog.manifest_sha256,
                    effective_config=configuration_dict(self.cfg))

    def _build_training_contract(self):
        sections = ('control', 'sim', 'asset', 'normalization', 'noise', 'domain_rand', 'navigation', 'evaluation', 'rewards', 'commands', 'camera', 'init_state')
        for role, catalog in self.catalogs.items():
            if sha256(catalog.path) != self.pool_identity['roles'][role]['sha256']:
                raise RuntimeError('Motor task manifest changed after loading')
        if sha256(self.project_root / self.cfg.task_catalog.pool_root / 'generation.json') != self.pool_identity['generation_sha256']:
            raise RuntimeError('Task generation identity changed after loading')
        robot_path = Path(self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=str(self.project_root))).resolve()
        assets = {'robot_urdf': sha256(robot_path)}
        for element in ET.parse(str(robot_path)).iter('mesh'):
            filename = element.attrib['filename']
            if filename.startswith('package://'):
                # This project's URDF package is the robots/Rotunbot directory.
                parts = filename[len('package://'):].split('/', 1)
                mesh_path = robot_path.parent.parent / parts[1]
            else:
                mesh_path = robot_path.parent / filename
            assets[filename] = sha256(mesh_path.resolve())
        shared_dir = Path(__file__).parent.parent / 'motor_sru'
        return dict(schema=1, task='direct_motor_trainable_depth_rl', profile=self.cfg.experiment_profile,
                    pool_identity=copy.deepcopy(self.pool_identity),
                    scene_layout=dict(type='centered_grid', spacing_m=float(self.cfg.env.env_spacing)),
                    robot_assets_sha256=assets,
                    config={name: configuration_dict(getattr(self.cfg, name)) for name in sections},
                    observations=dict(actor=2943, critic=188, proprio=380, proprio_frame=19, history=20,
                                      privileged_history=63, privileged_extra=125, raw_depth=2560,
                                      raw_depth_slice=[380,2940], actual_target_columns=[2940,2941], fresh_column=2942,
                                      body_action_history='original equivalent raw units: [3*a0,.9*a1]',
                                      actual_targets='previous applied rate-limited targets divided by [3,.45]',
                                      position_frame='fixed map axes; simulator isolation translation removed',
                                      privileged_pillars='dx,dy in robot yaw frame, radius; 40 pillars',
                                      privileged_boundaries='left,right,bottom,top map-boundary surface distances',
                                      remaining_time_column=187,
                                      remaining_time='clamp((9000-episode_length)/9000,0,1); reset returns 1'),
                    actions=dict(policy='finite normalized [-1,1]', target_scales=[3.,.45],
                                 mapping='original R executor applies scaling once, target limits, rate limits, then 35/300/150 torque law'),
                    frequencies=dict(physics_hz=50, motor_policy_hz=50, torque_controller_hz=50, depth_hz=5),
                    success=dict(distance_m=.4, speed_m_s=.1, duration_s=180., stop_dwell_s=0.),
                    radius_enforcement='none',
                    depth=dict(frozen_encoder=False, representation='raw axial metric depth in meters; invalid=0',
                               camera_mount='full base_link pose including actual roll/pitch/yaw'),
                    reward=dict(schema=self.cfg.rewards.reward_schema, scales=configuration_dict(self.cfg.rewards.scales),
                                failure_priority=['collision','bounds','unstable'],
                                progress='2*(previous_distance-current_distance); no sign or stationary reward'),
                    source_sha256={name: sha256(Path(__file__).parent / name) for name in ('env.py','config.py')},
                    shared_sources_sha256={name: sha256(shared_dir / name) for name in ('camera_pose.py','task_catalog.py','pillar.urdf')})

    def get_training_contract(self):
        current = self._build_training_contract()
        if current != self._training_contract:
            raise RuntimeError('Motor environment config/source/assets changed after construction')
        return copy.deepcopy(current)

    def get_checkpoint_state(self):
        return dict(catalog_sha256=self.catalog.manifest_sha256, sampler=self.sampler.state_dict(),
                    max_stage=self.cfg.task_catalog.max_stage, task_indices=self._task_indices_cpu.tolist())

    def set_checkpoint_state(self, state):
        if state['catalog_sha256'] != self.catalog.manifest_sha256:
            raise ValueError('Sampler checkpoint uses a different active catalog')
        if state.get('max_stage') != self.cfg.task_catalog.max_stage:
            raise ValueError('Cannot resume an environment with a different stage filter')
        self.sampler.load_state_dict(state['sampler'])
        indices = np.asarray(state['task_indices'], dtype=np.int64)
        if len(indices) != self.num_envs or any(int(i) not in self.sampler.eligible for i in indices):
            raise ValueError('Saved task indices are incompatible with the environment')
        self._task_indices_cpu[:] = indices
        self._reset_to_tasks(torch.arange(self.num_envs, device=self.device), resample=False)
