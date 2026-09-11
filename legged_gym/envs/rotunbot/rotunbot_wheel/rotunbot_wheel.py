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

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .rotunbot_wheel_config import RotunbotWheelCfg

class RotunbotWheel(LeggedRobot):
    '''
    Rotunbot is a class that represents a custom environment for a spherical robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        feet_height (torch.Tensor): Tensor representing the height of the feet.
        sim (gymtorch.GymSim): The simulation object.
        terrain (Terrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        _resample_commands():
        _get_phase(): Calculates the phase of the gait cycle.
        _get_stance_mask(): Calculates the gait phase.
        compute_ref_state(): Computes the reference state.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''
    cfg : RotunbotWheelCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
         # 添加历史状态存储
        self.last_base_lin_vel = torch.zeros_like(self.base_lin_vel)
        self.last_base_ang_vel = torch.zeros_like(self.base_ang_vel)
        # self.last_torques = torch.zeros((self.num_envs, self.num_actions), device=self.device)  # 存储上一时刻力矩
        self.step_counter = 0
        self.print_interval = 5  # 每50步打印一次
        self.data_print = True

    def compute_observations(self):
        """ Computes observations
        """
         # 计算速度差 (当前速度 - 上一时刻速度)
        lin_vel_diff = self.base_lin_vel - self.last_base_lin_vel  # 线速度差
        ang_vel_diff = self.base_ang_vel - self.last_base_ang_vel # 角速度差

        # self.obs_buf = torch.cat((  self.commands[:, :2],
        #                             # self.base_quat,
        #                             # self.root_states[:, 7:10],
        #                             # self.root_states[:, 10:13],
        #                             self.base_lin_vel, # 当前球坐标系线速度
        #                             self.base_ang_vel, # 当前球坐标系角速度
        #                             self.dof_pos[:,1].unsqueeze(1) ,
        #                             self.dof_vel,
        #                             self.actions,
        #                             self.last_base_lin_vel,         # 上一时刻线速度
        #                             self.last_base_ang_vel,         # 上一时刻角速度    
        #                             self.last_actions,              
        #                             self.last_dof_vel
        #                             # self.last_torques,              # [2] 上一时刻力矩
        #                             # self.last_dof_vel,
        #                             ),dim=-1)
        self.obs_buf = torch.cat((  self.commands[:, :2],     # [2] 速度指令
                                    self.base_quat,           # [4] 当前四元数
                                    # self.root_states[:, 7:10],
                                    # self.root_states[:, 10:13],
                                    self.base_lin_vel, # 当前球坐标系线速度 # [3]
                                    self.base_ang_vel, # 当前球坐标系角速度 # [3]
                                    self.dof_pos[:,1].unsqueeze(1) ,  # [1] 关节角度
                                    self.dof_vel[:,:2],                # [2] 关节角速度
                                    self.actions,                     # [3] 力矩
                                    # self.last_base_lin_vel,         # 上一时刻线速度
                                    # self.last_base_ang_vel,         # 上一时刻角速度    
                                    lin_vel_diff,                     # [3] 线速度差
                                    ang_vel_diff,                     # [3] 角速度差
                                    self.last_actions,                # [3] 上一时刻力矩           
                                    self.last_dof_vel[:,:2]            # [2] 上一时刻关节角速度
                                    # self.last_torques,              
                                    # self.last_dof_vel,
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
    
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            # torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
            torques = 5*(actions_scaled - self.dof_vel) - 0.5*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt 
            # print(torques)
        elif control_type=="T":
            torques = actions_scaled
            # print(torques)
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        torques[:,0] = torch.clip(torques[:,0], -self.cfg.control.torque_limits_1, self.cfg.control.torque_limits_1)
        torques[:,1] = torch.clip(torques[:,1], -self.cfg.control.torque_limits_2, self.cfg.control.torque_limits_2)
        torques[:,2] = torch.clip(torques[:,2], -self.cfg.control.torque_limits_3, self.cfg.control.torque_limits_3)
        return torques
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:2] = 0. # commands
        # noise_vec[2:6] = noise_scales.quat * noise_level
        # noise_vec[2:5] = noise_scales.lin_vel * noise_level
        # noise_vec[5:8] = noise_scales.ang_vel * noise_level
        # noise_vec[8] = noise_scales.dof_pos * noise_level
        # noise_vec[9:11] = noise_scales.dof_vel * noise_level
        # noise_vec[11:13] = 0. # previous actions
        # noise_vec[13:16] = noise_scales.lin_vel * noise_level
        # noise_vec[16:19] = noise_scales.ang_vel * noise_level
        # noise_vec[19:21] = 0. 
        # noise_vec[21:23] = noise_scales.dof_vel * noise_level
        noise_vec[2:6] = noise_scales.quat * noise_level
        noise_vec[6:9] = noise_scales.lin_vel * noise_level
        noise_vec[9:12] = noise_scales.ang_vel * noise_level
        noise_vec[12] = noise_scales.dof_pos * noise_level
        noise_vec[13:15] = noise_scales.dof_vel * noise_level
        noise_vec[15:18] = 0. # previous actions
        noise_vec[18:21] = noise_scales.lin_vel * noise_level
        noise_vec[21:24] = noise_scales.ang_vel * noise_level
        noise_vec[24:27] = 0.
        noise_vec[27:29] = noise_scales.ang_vel * noise_level

        return noise_vec
    
    

    def _process_dof_props(self, props, env_id):
        props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        props["stiffness"].fill(0.0)
        props["damping"].fill(0.0)
        return props

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # set small commands to zero
        # self.commands[env_ids, 0] *= (torch.norm(self.commands[env_ids, 0], dim=1) > 0.2).unsqueeze(1)
        for i in range(len(env_ids)):
            if self.commands[env_ids[i], 0] <= 0.1 and self.commands[env_ids[i], 0] >= -0.1:
                self.commands[env_ids[i], 0] = 0
        yaw_limit = torch.abs(self.commands[env_ids, 0]/4)
        for i in range(len(env_ids)):
            if self.commands[env_ids[i], 1] > yaw_limit[i]:
                self.commands[env_ids[i], 1] = yaw_limit[i]
            if self.commands[env_ids[i], 1] < -yaw_limit[i]:
                self.commands[env_ids[i], 1] = -yaw_limit[i]

    #------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # if self.step_counter % self.print_interval == 0:
        #     print(f"给定线速度: {self.commands[0, 0]:.4f}")  # 只打印第一项
        #     print(f"实际线速度y: {self.base_lin_vel[0, 1]:.4f}")  # 只打印第一项
        #     print(f"实际线速度x: {self.base_lin_vel[0, 0]:.4f}")  # 只打印第一项       
        # lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0]) + torch.square(self.base_lin_vel[:, 1])
        # return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
        e = 1e-3
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])/(abs(self.commands[:, 0])+e) + torch.square(self.base_lin_vel[:, 1])
        if self.step_counter % self.print_interval == 0 and self.data_print:
            print(f"给定线速度: {self.commands[0, 0]:.4f}")  # 只打印第一项
            print(f"实际线速度y: {self.base_lin_vel[0, 1]:.4f}")  # 只打印第一项
            print(f"实际线速度x: {self.base_lin_vel[0, 0]:.4f}")  # 只打印第一项
            print(f"输出力矩: {self.torques[0, 0]:.4f}, {self.torques[0, 1]:.4f}, {self.torques[0, 2]:.4f}")         
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
        # return 1
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        # if self.step_counter % self.print_interval == 0:
        #     print(f"给定角速度: {self.commands[0, 1]:.4f}")  # 只打印第一项
        #     print(f"实际角速度: {self.base_ang_vel[0, 2]:.4f}")  # 只打印第一项
        #     print('**********************')
        # ang_vel_error = torch.square(self.commands[:, 1] - self.base_ang_vel[:, 2])
        # return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
        e = 1e-3
        if self.step_counter % self.print_interval == 0 and self.data_print:
            print(f"给定角速度: {self.commands[0, 1]:.4f}")  # 只打印第一项
            print(f"实际角速度: {self.base_ang_vel[0, 2]:.4f}")  # 只打印第一项
            print('**********************')
        ang_vel_error = torch.square((self.commands[:, 1] - self.base_ang_vel[:, 2])/(abs(self.commands[:, 1])+e))
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
        # return 1
    def step(self, actions):
        self.step_counter += 1
        # self.last_base_lin_vel = self.root_states[:, 7:10].clone()  # 复制当前线速度
        # self.last_base_ang_vel = self.root_states[:, 10:13].clone() # 复制当前角速度
        # self.last_torques = self.torques.clone()  
        self.last_base_lin_vel = self.base_lin_vel.clone()  # 复制当前线速度
        self.last_base_ang_vel = self.base_ang_vel.clone() # 复制当前角速度
        return super().step(actions)
    
    def reset_idx(self, env_ids):
        # 重置时也要重置历史信息
        self.last_base_lin_vel[env_ids] = 0
        self.last_base_ang_vel[env_ids] = 0
        # self.last_torques[env_ids] = 0
        
        super().reset_idx(env_ids)
