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
from collections import deque

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .rotunbot_real_config import RotunbotRealCfg
from scipy.spatial.transform import Rotation

class RotunbotReal(LeggedRobot):
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
    cfg : RotunbotRealCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
         # 添加历史状态存储
        self.last_base_lin_vel = torch.zeros_like(self.base_lin_vel)
        self.last_base_ang_vel = torch.zeros_like(self.base_ang_vel)
        self.step_counter = 0
        self.print_interval = 5  # 每50步打印一次
        real_network_path = 'legged_gym/envs/rotunbot/vel_tracking/model_jit.pth'
        self.real_network = torch.jit.load(real_network_path).to(self.device)

    def compute_observations(self):
        """ Computes observations
        """
         # 计算速度差 (当前速度 - 上一时刻速度)
        lin_vel_diff = self.root_states[:, 7:10] - self.last_base_lin_vel  # 线速度差
        ang_vel_diff = self.root_states[:, 10:13] - self.last_base_ang_vel # 角速度差
        base_ret = Rotation.from_quat(self.base_quat.cpu().numpy())
        self.base_euler_tensor = torch.as_tensor(base_ret.as_euler('xyz'),dtype=torch.float,device=self.device)
        # print(self.commands[:, :2].shape)
        self.single_obs_buf = torch.cat((  
                                    self.base_euler_tensor,
                                    self.base_lin_vel,
                                    self.base_ang_vel,
                                    self.dof_pos[:,1].unsqueeze(1) ,
                                    self.dof_vel[:,0].unsqueeze(1) , 
                                    ),dim=-1)
        self.obs_history.append(self.single_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K
        self.obs_real = self.real_network(obs_buf_all)

        self.obs_buf = torch.cat((  self.commands[:, :2],#*10,
                                    self.obs_real,
                                    self.actions,
                                     
                                    ),dim=-1)
        # print(self.obs_buf[0])
        # add perceptive inputs if not blind
        # print(self.obs_buf.shape)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        # if self.add_noise:
        #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
    
    
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            # 假设 actions[:, 0] 是关节1的目标旋转角速度
            # 假设 actions[:, 1] 是关节2的目标位置
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            
            # 创建目标张量
            vel_targets = torch.zeros((self.num_envs, self.num_dof), device=self.device)
            pos_targets = torch.zeros((self.num_envs, self.num_dof), device=self.device)
            
            # 设置目标值
            vel_targets[:, 0] = self.torques[:, 0]  # 第一个关节的速度目标
            pos_targets[:, 1] = self.torques[:, 1]  # 第二个关节的位置目标
            
            # 使用张量API设置目标
            self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(vel_targets))
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_targets))

            # 进行物理仿真
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        self.last_base_lin_vel = self.base_lin_vel.clone()  # 复制当前线速度
        self.last_base_ang_vel = self.base_ang_vel.clone() # 复制当前角速度 2月21号 16.14
        self.last_obs_real = self.obs_real.clone()  # 复制当前观测值
        # ... existing code ...

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

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
        elif control_type == "S_P":
            # 对每个环境的第一个关节乘上1，第二个关节乘上0.4
            actions_scaled = actions.clone()  # 复制原始动作
            actions_scaled[:, 0] *= 1.0  # 第一个关节乘上1
            actions_scaled[:, 1] *= 0.5  # 第二个关节乘上0.4
            
            # 对torques进行限幅
            actions_scaled[:, 0] = torch.clamp(actions_scaled[:, 0], -6.0, 6.0)  # 限制第一个关节在正负25之间
            actions_scaled[:, 1] = torch.clamp(actions_scaled[:, 1], -0.5236, 0.5236)  # 限制第二个关节在正负30du之间
            
            torques = actions_scaled  # 使用调整后的动作
            if self.step_counter % self.print_interval == 0:
                print(f"第一个关节动作: {actions_scaled[0, 0]:.4f}")
                print(f"第二个关节动作: {actions_scaled[0, 1]:.4f}")
                print('**********************')
            return torques
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
    
    def _init_buffers(self):
        super()._init_buffers()
        self.obs_real = torch.zeros(self.num_envs, self.cfg.env.num_observations-4, dtype=torch.float, device=self.device)
        self.last_obs_real = torch.zeros(self.num_envs, self.cfg.env.num_observations-4, dtype=torch.float, device=self.device)
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.cfg.env.num_observations-4, dtype=torch.float, device=self.device))
    
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        # # 第一个关节使用速度控制
        props["driveMode"][0] = gymapi.DOF_MODE_VEL
        # 第二个关节使用位置控制
        props["driveMode"][1] = gymapi.DOF_MODE_POS
        
        # 设置PD控制器参数
        props["stiffness"][0] = 0.0  # 速度控制不需要刚度
        props["damping"][0] = 35.0    # 速度控制的阻尼
        
        props["stiffness"][1] = 300  # 位置控制的刚度
        props["damping"][1] = 150    # 位置控制的阻尼
        
        return props
    
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
        noise_vec[2:6] = noise_scales.quat * noise_level
        noise_vec[6:9] = noise_scales.lin_vel * noise_level
        noise_vec[9:12] = noise_scales.ang_vel * noise_level
        # noise_vec[12] = noise_scales.dof_pos * noise_level
        # noise_vec[13:15] = noise_scales.dof_vel * noise_level
        # noise_vec[15:17] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[18:205] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec
    
    

    # def _process_dof_props(self, props, env_id):
    #     # props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
    #     # props["stiffness"].fill(0.0)
    #     # props["damping"].fill(0.0)
    #     props["driveMode"].fill(gymapi.DOF_MODE_POS)
    #     props["stiffness"].fill(1000.0)
    #     props["damping"].fill(200.0)
    #     return props

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # set small commands to zero
        # self.commands[env_ids, 0] *= (torch.norm(self.commands[env_ids, 0], dim=1) > 0.2).unsqueeze(1)
        yaw_limit = torch.abs(self.commands[env_ids, 0]/1.4)
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
        # lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 1]) + torch.square(self.base_lin_vel[:, 0])
        # return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
        e = 1e-3
        lin_vel_error = torch.square(self.commands[:, 0] - self.obs_real[:, 3])/(abs(self.commands[:, 0])+e) + torch.square(self.obs_real[:, 3])
        if self.step_counter % self.print_interval == 0:
            print(f"给定线速度: {self.commands[0, 0]:.4f}")  # 只打印第一项
            print(f"实际线速度y: {self.obs_real[0, 3]:.4f}")  # 只打印第一项
            print(f"实际线速度x: {self.obs_real[0, 4]:.4f}")  # 只打印第一项    
            print(f"实际线速度z: {self.obs_real[0, 5]:.4f}")     
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
        # return 1
    
    # def _reward_tracking_ang_vel(self):
    #     # Tracking of angular velocity commands (yaw) 
    #     # if self.step_counter % self.print_interval == 0:
    #     #     print(f"给定角速度: {self.commands[0, 1]:.4f}")  # 只打印第一项
    #     #     print(f"实际角速度: {self.base_ang_vel[0, 2]:.4f}")  # 只打印第一项
    #     #     print('**********************')
    #     # ang_vel_error = torch.square(self.commands[:, 1] - self.base_ang_vel[:, 2])
    #     # return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    #     e = 1e-3
    #     ang_vel_error = torch.square(self.commands[:, 1] - self.obs_real[:, 8])
    #     # if self.step_counter % self.print_interval == 0:
    #     #     print(f"给定角速度: {self.commands[0, 1]:.4f}")  # 只打印第一项
    #     #     print(f"实际角速度x: {self.base_ang_vel[0, 0]:.4f}")
    #     #     print(f"实际角速度y: {self.base_ang_vel[0, 1]:.4f}")
    #     #     print(f"实际角速度z: {self.base_ang_vel[0, 2]:.4f}")  # 只打印第一项
    #     #     print('**********************')
    #     ang_vel_error = torch.square((self.commands[:, 1] - self.obs_real[:, 8])/(abs(self.commands[:, 1])+e))
    #     return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    #     return 1

    def _reward_tracking_ang_vel(self):
        e = 1e-3
        ang_vel_error = torch.square((self.commands[:, 1] - self.obs_real[:, 8])/(abs(self.commands[:, 1])+e))
        ang_vel_smooth = torch.square(self.obs_real[:, 8] - self.last_obs_real[:, 8])  # 添加平滑项
        if self.step_counter % self.print_interval == 0:
            print(f"给定角速度: {self.commands[0, 1]:.4f}")  # 只打印第一项
            print(f"实际角速度x: {self.obs_real[0, 6]:.4f}")
            print(f"实际角速度y: {self.obs_real[0, 7]:.4f}")
            print(f"实际角速度z: {self.obs_real[0, 8]:.4f}")  # 只打印第一项
            print('**********************')
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma) - 0.1 * ang_vel_smooth


    # def _reward_tracking_ang_vel(self):
    #     e = 1e-3
    #     # 计算角速度误差
    #     ang_vel_error = torch.square((self.commands[:, 1] - self.base_ang_vel[:, 2])/(abs(self.commands[:, 1])+e))
        
    #     # 计算角速度方向奖励（当实际角速度方向与目标方向一致时给予额外奖励）
    #     direction_reward = torch.sign(self.commands[:, 1]) * torch.sign(self.base_ang_vel[:, 2])
    #     direction_reward = torch.clamp(direction_reward, 0.0, 1.0)  # 将方向奖励限制在[0,1]范围内
        
    #     # 计算角速度平滑项（降低权重以减少对跟踪性能的影响）
    #     ang_vel_smooth = torch.square(self.base_ang_vel[:, 2] - self.last_base_ang_vel[:, 2])
        
    #     # 计算角速度大小奖励（当实际角速度接近目标时给予额外奖励）
    #     magnitude_reward = torch.exp(-torch.abs(self.commands[:, 1] - self.base_ang_vel[:, 2])/0.5)
        
    #     # 添加主轴动作变化惩罚项
    #     if not hasattr(self, 'last_actions'):
    #         self.last_actions = torch.zeros_like(self.actions)
    #     action_change = torch.abs(self.actions[:, 0] - self.last_actions[:, 0])  # 主轴关节动作变化
    #     action_smooth_penalty = torch.exp(-action_change/0.3)  # 动作变化越大，惩罚越大
    #     self.last_actions = self.actions.clone()
        
    #     if self.step_counter % self.print_interval == 0:
    #         print(f"给定角速度: {self.commands[0, 1]:.4f}")
    #         print(f"实际角速度x: {self.base_ang_vel[0, 0]:.4f}")
    #         print(f"实际角速度y: {self.base_ang_vel[0, 1]:.4f}")
    #         print(f"实际角速度z: {self.base_ang_vel[0, 2]:.4f}")
    #         print(f"方向奖励: {direction_reward[0
        # self.last_obs_real[env_ids] = 0]:.4f}")
    #         print(f"大小奖励: {magnitude_reward[0]:.4f}")
    #         print(f"主轴动作变化: {action_change[0]:.4f}")
    #         print(f"动作平滑惩罚: {action_smooth_penalty[0]:.4f}")
    #         print('**********************')
        
    #     # 组合所有奖励项
    #     tracking_reward = torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    #     final_reward = tracking_reward + 0.3 * direction_reward + 0.2 * magnitude_reward - 0.05 * ang_vel_smooth + 0.2 * action_smooth_penalty
        
    #     return final_reward

    
    def reset_idx(self, env_ids):
        # 重置时也要重置历史信息
        self.last_base_lin_vel[env_ids] = 0
        self.last_base_ang_vel[env_ids] = 0
        self.last_obs_real[env_ids] = 0
        self.obs_history.clear()
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.cfg.env.num_observations-4, dtype=torch.float, device=self.device))
        
        super().reset_idx(env_ids)
