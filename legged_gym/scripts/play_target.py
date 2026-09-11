# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import csv

"""
球形机器人端到端自动导航play函数
输入要到达的目标点和到达目标点的方向(yaw?)
输入cfg : rotunbot_target
"""
def play(args):
    args.task = 'rotunbot_target'
    # args.task = 'rotunbot_target_obstacle'
    # args.load_run = '/home/an/legged_gym/logs/rotunbot_target/Apr15_00-26-59_'
    # args.load_run = '/home/an/legged_gym/logs/rotunbot_target/Apr15_23-14-37_'
    # args.load_run = '/home/an/legged_gym/logs/rotunbot_target/Apr22_10-50-27_' # 正常（距离0.1）
    # args.load_run = '/home/an/legged_gym/logs/rotunbot_target/Apr22_19-45-47_'  #正常,球会绕yaw轴自传（距离0.06）
    # args.load_run = '/home/an/legged_gym/logs/rotunbot_target/Apr27_00-02-03_'  #命令包括角度（效果很差）
    # args.load_run = '/home/an/legged_gym/logs/rotunbot_target/Apr27_20-34-20_'  #命令包括角度（效果很差）
    # args.load_run = '/home/an/legged_gym/logs/rotunbot_target/Apr29_22-28-11_' #命令不包括角度 
    # args.load_run = '/home/an/legged_gym/logs/rotunbot_target/May08_23-24-55_' # 不正常（距离0.08）
    # args.load_run = '/home/an/legged_gym/logs/rotunbot_target_obstacle/May12_21-47-36_'   #
    # args.load_run = '/home/an/legged_gym/logs/rotunbot_target/May20_23-45-55_'  #正常（距离0.06
    # args.load_run = '/home/an/legged_gym/logs/rotunbot_target/Jul04_23-28-38_' # 球会绕yaw轴自传
    # args.load_run = '/home/an/legged_gym/logs/rotunbot_target/Jul25_10-51-54_' #还行
    args.load_run = '/home/an/legged_gym/logs/rotunbot_target/Aug02_23-57-43_' # 
    # args.load_run = '/home/an/legged_gym/logs/rotunbot_target/Aug28_14-17-26_'
    # args.load_run = '/home/an/legged_gym/logs/rotunbot_target/Aug29_13-01-29_' #还行
    # args.load_run = '/home/an/legged_gym/logs/rotunbot_target/Sep03_11-05-19_' #还行
    args.load_run = '/home/an/legged_gym/logs/rotunbot_target/Sep08_12-19-05_'
    


    # args.task = 'rotunbot_vel'
    # args.load_run = '/home/an/legged_gym/logs/rotunbot_vel/Aug24_12-34-19_' # 

    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    data_record = False # whether to record data
    Torque_Output_index = np.zeros(21)

    if data_record:
        file_path = 'data/obs_record.csv'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        f=open(file_path,'w',encoding='utf-8')
        writer = csv.writer(f)
        writer.writerow(['torque1','torque2','v1','v2','pos1','pos2','quat_x','quat_y','quat_z','quat_w','lin_vel_x','lin_vel_y','lin_vel_z','ang_vel_x','ang_vel_y','ang_vel_z','pos_x','pos_y','euler_x','euler_y','euler_z'])


    # override some parameters for testing
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    train_cfg.seed = 656

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index1 = 0 # which joint is used for logging
    joint_index2 = 1
    stop_state_log = 3200 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, critic_obs , rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index1].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, 1].item(),
                    'dof_vel': env.dof_vel[robot_index, 0].item(),
                    'dof_torque1': env.torques[robot_index, joint_index1].item(),
                    'dof_torque2': env.torques[robot_index, joint_index2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    'base_pos_x': env.root_states[robot_index, 0].item(),
                    'base_pos_y': env.root_states[robot_index, 1].item(),
                    'base_vel_z': env.root_states[robot_index, 2].item(),
                    'ref_pos_x': env.commands[robot_index, 0].item(),
                    'ref_pos_y': env.commands[robot_index, 1].item(),
                    'goal_dist':env.goal_dist[robot_index].item()
                }
            )
            if data_record:
                Torque_Output_index[0] = env.torques[robot_index, 0].item()
                Torque_Output_index[1] = env.torques[robot_index, 1].item()
                Torque_Output_index[2] = env.dof_vel[robot_index, 0].item()
                Torque_Output_index[3] = env.dof_vel[robot_index, 1].item()
                Torque_Output_index[4] = env.dof_pos[robot_index, 0].item()
                Torque_Output_index[5] = env.dof_pos[robot_index, 1].item()
                Torque_Output_index[6] = env.base_quat[robot_index, 0].item()
                Torque_Output_index[7] = env.base_quat[robot_index, 1].item()
                Torque_Output_index[8] = env.base_quat[robot_index, 2].item()
                Torque_Output_index[9] = env.base_quat[robot_index, 3].item()
                Torque_Output_index[10] = env.base_lin_vel[robot_index, 0].item()
                Torque_Output_index[11] = env.base_lin_vel[robot_index, 1].item()
                Torque_Output_index[12] = env.base_lin_vel[robot_index, 2].item()
                Torque_Output_index[13] = env.base_ang_vel[robot_index, 0].item()
                Torque_Output_index[14] = env.base_ang_vel[robot_index, 1].item()
                Torque_Output_index[15] = env.base_ang_vel[robot_index, 2].item()
                Torque_Output_index[16] = env.root_states[robot_index, 0].item()
                Torque_Output_index[17] = env.root_states[robot_index, 1].item()
                Torque_Output_index[18] = env.base_euler_tensor[robot_index, 0].item()
                Torque_Output_index[19] = env.base_euler_tensor[robot_index, 1].item()
                Torque_Output_index[20] = env.base_euler_tensor[robot_index, 2].item()
                writer.writerow(Torque_Output_index)
        elif i==stop_state_log:
            logger.plot_trajectories()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
