"""CPU checks of real environment methods; no Isaac Gym or GPU initialization."""
import ast
import copy
import hashlib
import json
import math
import os
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[5]
SOURCE = Path(__file__).resolve().parent


def load_methods():
    namespace = dict(np=np, torch=torch, copy=copy, hashlib=hashlib, json=json, math=math, os=os,
                     Path=Path, SimpleNamespace=SimpleNamespace, ET=ET, deque=deque,
                     RotunbotTargetRepro=object, __file__=str(SOURCE / 'env.py'),
                     gymtorch=SimpleNamespace(unwrap_tensor=lambda x: x))
    old_catalog = ast.parse((SOURCE.parent / 'motor_sru' / 'task_catalog.py').read_text(encoding='utf-8'))
    functions = [node for node in old_catalog.body if isinstance(node, ast.FunctionDef) and node.name in ('sha256', 'geometry_hash')]
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<catalog helpers>', 'exec'), namespace)
    tree = ast.parse((SOURCE / 'env.py').read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE / 'env.py'), 'exec'), namespace)
    config_tree = ast.parse((SOURCE / 'config.py').read_text(encoding='utf-8'))
    config_class = next(node for node in config_tree.body if isinstance(node, ast.ClassDef) and node.name == 'MotorDepthCfg')
    rewards = copy.deepcopy(next(node for node in config_class.body if isinstance(node, ast.ClassDef) and node.name == 'rewards'))
    rewards.bases = []
    exec(compile(ast.Module(body=[rewards], type_ignores=[]), '<actual reward config>', 'exec'), namespace)
    return namespace


def check_rewards(ns):
    n = 6
    zeros = torch.zeros(n)
    previous = torch.tensor([4., 4., 4., 4., 4., .3])
    distance = torch.tensor([4., 3.9, 4.1, 4., 4., .3])
    success = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.bool)
    collision = torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.bool)
    bounds = torch.tensor([0, 0, 0, 1, 1, 0], dtype=torch.bool)
    unstable = torch.tensor([0, 0, 0, 1, 1, 0], dtype=torch.bool)
    total, terms = ns['reward_terms'](previous, distance, success, collision, bounds, unstable,
                                    torch.zeros(n, 2), zeros, torch.zeros(n, 2), torch.zeros(n, 2),
                                    torch.zeros(n, 2), .02, ns['rewards'])
    assert torch.allclose(total[:3], torch.tensor([-.0004, .1996, -.2004]), atol=1e-6)
    assert abs((terms['progress_per_m'][1] + terms['progress_per_m'][2]).item()) < 1e-6
    assert torch.allclose(total[3:5], torch.tensor([-10.0004, -10.0004]))
    assert abs(total[5].item()-39.9996) < 1e-5
    assert set(terms) == set(key for key in vars(ns['rewards'].scales) if not key.startswith('_'))
    assert sum(terms.values()).equal(total)
    return dict(stationary_per_step=total[0].item(), advance_10cm=total[1].item(), retreat_10cm=total[2].item(),
                success=total[5].item(), combined_failure_penalty=total[3].item(), inherited_terms_absent=True)


def check_original_executor(ns):
    old_path = SOURCE.parent / 'envs' / 'rotunbot' / 'target_point' / 'rotunbot_target_lh.py'
    tree = ast.parse(old_path.read_text(encoding='utf-8'))
    old_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RotunbotTargetLH')
    torque_method = next(node for node in old_class.body if isinstance(node, ast.FunctionDef) and node.name == '_compute_torques')
    temp = dict(torch=torch)
    exec(compile(ast.Module(body=[torque_method], type_ignores=[]), str(old_path), 'exec'), temp)
    control = SimpleNamespace(action_scale=40, control_type='R', first_actionScale=3., second_actionScale=.45,
                              first_vel_limits=3., second_pos_limits=.45, set_a_rate_limit=False,
                              rate_limit_1=.02, rate_limit_2=.04, torque_limits_1=100., torque_limits_2=100.)
    env = ns['MotorDepthEnv'].__new__(ns['MotorDepthEnv'])
    env.cfg = SimpleNamespace(control=control)
    env.device = 'cpu'
    env.dof_pos = torch.zeros(2, 2)
    env.dof_vel = torch.zeros(2, 2)
    env.last_output_actions = torch.zeros(2, 2)
    env.actions = torch.tensor([[1., -1.], [-1., 1.]])
    torques = temp['_compute_torques'](env, env.actions)
    assert torch.allclose(env.output_actions, torch.tensor([[3., -.45], [-3., .45]]))
    assert torques.abs().max() <= 100.
    assert torch.allclose(env._legacy_raw_actions(), torch.tensor([[3., -.9], [-3., .9]]))
    control.set_a_rate_limit = True
    env.last_output_actions.zero_()
    temp['_compute_torques'](env, env.actions)
    assert torch.allclose(env.output_actions, torch.tensor([[.02, -.04], [-.02, .04]]))
    assert torch.allclose(env._normalized_motor_targets(), env.output_actions / torch.tensor([3., .45]))
    max_delta = torch.zeros(2)
    for index in range(400):
        env.last_output_actions = env.output_actions.clone()
        command = env.actions if index < 200 else -env.actions
        temp['_compute_torques'](env, command)
        delta = (env.output_actions-env.last_output_actions).abs().amax(0)
        max_delta = torch.maximum(max_delta, delta)
        assert (delta <= torch.tensor([.02, .04])+1e-6).all()
        assert (env.output_actions.abs() <= torch.tensor([3., .45])+1e-6).all()
    return dict(full_scale_targets=[3., .45], original_legacy_action_units=[3., .9],
                maximum_target_delta=max_delta.tolist(), original_R_method_tested=True)


def check_catalog(ns, project_root):
    catalogs, identity = ns['validate_pool'](project_root, 'artifacts/MOTOR_SRU_INTEGRATION_20260906/depth_avoidance_redesign/maps')
    assert {role: len(catalog.tasks) for role, catalog in catalogs.items()} == dict(train=192, val=48, test=48, diagnostic=12)
    tasks = catalogs['train'].tasks
    sampler = ns['StageTaskSampler'](tasks, 60, 0)
    assert len(sampler.eligible) == 64
    draw = sampler.draw(64)
    assert len(set(draw.tolist())) == 64 and all(tasks[int(i)]['stage'] == 0 for i in draw)
    saved = copy.deepcopy(sampler.state_dict())
    expected = sampler.draw(139)
    restored = ns['StageTaskSampler'](tasks, 60, 0)
    restored.load_state_dict(saved)
    assert np.array_equal(restored.draw(139), expected)
    mixed = ns['StageTaskSampler'](tasks, 60, 1)
    assert len(mixed.eligible) == 128
    try:
        mixed.load_state_dict(saved)
    except ValueError:
        pass
    else:
        raise AssertionError('Changing the stage filter silently restored old sampling state')
    return catalogs, dict(counts={role: len(catalog.tasks) for role, catalog in catalogs.items()},
                          stage0_eligible=64, stage1_eligible=128, sampler_replay_exact=True, identity=identity)


def check_reset_and_observations(ns, catalogs):
    env = ns['MotorDepthEnv'].__new__(ns['MotorDepthEnv'])
    env.num_envs, env.device = 2, 'cpu'
    env.cfg = SimpleNamespace(task_catalog=SimpleNamespace(resample_on_reset=False),
                              normalization=SimpleNamespace(clip_observations=100.),
                              camera=SimpleNamespace(motor_steps_per_frame=10),
                              navigation=SimpleNamespace(pillar_count=40, pillar_radius=.4, robot_radius=.4))
    env.catalog = catalogs['diagnostic']
    env._task_indices_cpu = np.array([0, 1], dtype=np.int64)
    env.task_indices = torch.tensor([0, 1])
    env.task_ids = ['dirty0', 'dirty1']
    env.task_stages = torch.ones(2, dtype=torch.long)
    env.category_id = torch.ones(2, dtype=torch.long)
    env.task_starts, env.task_local_goals = torch.ones(2, 2), torch.ones(2, 2)
    env.pillar_centers, env.task_bounds = torch.ones(2, 40, 2), torch.ones(2, 4)
    env.env_origins = torch.tensor([[0., 0., 0.], [32., 0., 0.]])
    env._all_root_states = torch.ones(82, 13)
    env.root_states = env._all_root_states.view(2, 41, 13)[:, 0, :]
    env.base_quat = env.root_states[:, 3:7]
    env.base_init_state = torch.tensor([0., 0., .4, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
    env.robot_actor_indices = torch.tensor([0, 41], dtype=torch.int32)
    env.pillar_actor_indices = torch.stack((torch.arange(1, 41), torch.arange(42, 82)))
    env.default_dof_pos = torch.zeros(1, 2)
    env.dof_state = torch.ones(4, 2)
    env.dof_pos = env.dof_state.view(2, 2, 2)[..., 0]
    env.dof_vel = env.dof_state.view(2, 2, 2)[..., 1]
    env.commands = torch.ones(2, 3)
    env.base_lin_vel, env.base_ang_vel = torch.ones(2, 3), torch.ones(2, 3)
    env.gravity_vec = torch.tensor([[0., 0., -1.], [0., 0., -1.]])
    env.projected_gravity = torch.ones(2, 3)
    env.base_euler_tensor = torch.ones(2, 3)
    env._update_base_euler = lambda: env.base_euler_tensor.copy_(torch.stack((torch.zeros(2), torch.zeros(2), 2*torch.atan2(env.base_quat[:, 2], env.base_quat[:, 3])), dim=-1))
    for name in ('actions','last_actions','output_actions','last_output_actions','torques','last_dof_vel','_requested_raw_actions'):
        setattr(env, name, torch.ones(2, 2))
    for name in ('last_root_vel','last_base_lin_vel','last_base_ang_vel','last_root_states','feet_air_time','last_contacts'):
        setattr(env, name, torch.ones(2, 3))
    for name in ('goal_dist','last_goal_dist','orientation_error','last_orientation_error','episode_returns','actual_yaw_rate','previous_yaw','terminal_goal_dist','terminal_speed','terminal_balance_reward'):
        setattr(env, name, torch.ones(2))
    for name in ('success_buf','arrived_target_buf','stop_buf','collision_buf','physical_pillar_contact','bounds_buf','unstable_buf','terminal_timeout','terminal_unstable','terminal_out_of_bounds','depth_fresh','reset_buf','time_out_buf'):
        setattr(env, name, torch.ones(2, dtype=torch.bool))
    env.terminal_position = torch.ones(2, 2)
    env.episode_length_buf = torch.ones(2, dtype=torch.long)
    env.task_reset_count = torch.zeros(2, dtype=torch.long)
    env.depth_image = torch.ones(2, 40, 64)
    env.depth_stats = torch.ones(2, 3)
    env.capture_camera_pose = torch.ones(2, 7)
    env.depth_age = torch.ones(2, dtype=torch.long)
    env._proprio_pending = torch.zeros(2, dtype=torch.bool)
    env._depth_pending = torch.zeros(2, dtype=torch.bool)
    env._all_contact_forces = torch.ones(2, 44, 3)
    env.obs_history = deque([torch.ones(2, 19) for _ in range(20)], maxlen=20)
    env.critic_history = deque([torch.ones(2, 21) for _ in range(3)], maxlen=3)
    env.episode_sums = {'progress_per_m': torch.ones(2)}
    env.sim = None
    indexed_calls = []
    env.gym = SimpleNamespace(set_actor_root_state_tensor_indexed=lambda *args: indexed_calls.append(args[-2].clone()),
                              set_dof_state_tensor_indexed=lambda *args: None)
    untouched_root = env.root_states[1].clone()
    env._reset_to_tasks(torch.tensor([0]), resample=False)
    assert env.task_reset_count.tolist() == [1, 0]
    assert len(indexed_calls[0]) == 41 and set(indexed_calls[0].tolist()) == set(range(41))
    assert env.root_states[1].equal(untouched_root)
    for name in ('actions','last_actions','output_actions','last_output_actions','torques','_requested_raw_actions','depth_image','capture_camera_pose'):
        assert not getattr(env, name)[0].count_nonzero(), name
        assert getattr(env, name)[1].count_nonzero(), name
    assert all(not history[0].count_nonzero() and history[1].count_nonzero() for history in tuple(env.obs_history)+tuple(env.critic_history))
    assert env.depth_age[0].item() == 10 and env._depth_pending[0] and not env.depth_fresh[0]
    assert env.goal_dist[0] == env.last_goal_dist[0] and env.goal_dist[0] > .4
    assert torch.allclose(env.pillar_centers[0], torch.tensor(env.catalog.tasks[0]['posts']))
    env.compute_observations = lambda: None
    env.obs_buf = torch.zeros(2, 380)
    env.privileged_obs_buf = torch.zeros(2, 63)
    env.depth_image[0] = 2.5
    env.depth_fresh[:] = True
    env.output_actions[:] = torch.tensor([1.5, -.225])
    env.max_episode_length = 9000
    actor = env.get_observations()
    critic = env.get_privileged_observations()
    assert actor.shape == (2, 2943) and critic.shape == (2, 188)
    assert (actor[0, 380:2940] == 2.5).all()
    assert torch.allclose(actor[:, 2940:2942], torch.tensor([[.5, -.5], [.5, -.5]]))
    assert actor[:, 2942].eq(1.).all() and critic[0, 187] == 1.
    return dict(reset_cleared_raw_depth_actions_and_histories=True, other_environment_untouched=True,
                root_actors_per_reset=41, actor_shape=list(actor.shape), critic_shape=list(critic.shape),
                actual_targets_in_observation=True, no_frozen_encoder=True)


def main():
    ns = load_methods()
    project_root = SOURCE.parents[4]
    catalogs, catalog_result = check_catalog(ns, project_root)
    report = dict(status='PASS', reward=check_rewards(ns), executor=check_original_executor(ns),
                  catalogs=catalog_result, reset_and_observations=check_reset_and_observations(ns, catalogs))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
