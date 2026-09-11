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
from collections import deque

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict
import math

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .rotunbot_tra_config import RotunbotTrajectoryCfg

class RotunbotTrajectory(LeggedRobot):
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
    cfg : RotunbotTrajectoryCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
         # 添加历史状态存储
        self.num_single_obs = cfg.env.num_single_obs
        self.num_short_obs = int(cfg.env.num_single_obs * cfg.env.short_frame_stack)
        # self.last_torques = torch.zeros((self.num_envs, self.num_actions), device=self.device)  # 存储上一时刻力矩
        self.tra_step_counter = 0
        self.print_interval = 5  # 每50步打印一次
        self.data_print = True
        
    # def step(self, actions):
        
    #     return super().step(actions)
    
    def compute_observations(self):
        """ Computes observations
            DWL Version
                Priviledged_obs_buf(Critic_obs)
                obs_buf
        """
        # 加入位置信息，当前轨迹点
        # self.base_pos
        # self.Circle_Trajectory
        Tra_time_step = self.episode_length_buf // 5
        for i in range(self.num_envs):
            if(Tra_time_step[i] >= 350):
                Tra_time_step[i] = 349
        for i in range(self.num_envs):
            self.Tra_pos_now[i, :2] = torch.tensor(self.Circle_Trajectory[Tra_time_step[i]], device=self.device)
        
        self.tra_step_counter += 1
        if self.tra_step_counter % 5 ==0:
            self.tra_step_counter = 0
            self.Real_Trajectory.append(self.base_pos) 
            self.Real_Trajectory_Point = self.base_pos.clone()
        
        # 24
        privileged_obs_buf = torch.cat((
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 2
            self.dof_vel * self.obs_scales.dof_vel,  # 2
            self.actions,  # 2
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_quat * self.obs_scales.quat,  # 4
            self.rand_push_force[:, :2],  # 2
            self.env_frictions,  # 1
            self.Real_Trajectory_Point * self.obs_scales.pos,  # 3
            self.Tra_pos_now * self.obs_scales.pos,  # 2  
        ), dim=-1)

        #17
        obs_buf = torch.cat((  
                               self.base_quat,                  #4 
                               self.base_ang_vel,               #3
                               self.dof_pos[:, 1].unsqueeze(1), #1
                               self.dof_vel ,                   #2
                               self.actions ,                   #2
                               self.Real_Trajectory_Point,           #3  
                               self.Tra_pos_now,                       #2    
                               ),dim=-1)
        
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            obs_now = obs_buf.clone() + (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        
        self.obs_history.append(obs_now)
        self.critic_history.append(privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)
        self.root_states[:, 7:9] = self.rand_push_force[:, :2] # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

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
        actions_scaled = actions.clone().to(self.device)
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        elif control_type=="R":
            torques = actions.clone().to(self.device)
            actions_scaled[:,0] = actions[:,0] * self.cfg.control.first_actionScale
            actions_scaled[:,1] = actions[:,1] * self.cfg.control.second_actionScale
            print(actions_scaled)
            print(self.dof_pos)
            torques[:,0] =  25 * (actions_scaled[:,0] - self.dof_vel[:,0]) - 1 * (self.dof_vel[:,0] - self.last_dof_vel[:,0]) / self.dt
            torques[:,1] = 15 * ( actions_scaled[:,1]  - self.dof_pos[:,1]) - 5 * self.dof_vel[:,1]
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        torques[:,0] = torch.clip(torques[:,0], -self.cfg.control.torque_limits_1, self.cfg.control.torque_limits_1)
        torques[:,1] = torch.clip(torques[:,1], -self.cfg.control.torque_limits_2, self.cfg.control.torque_limits_2)
        return torques
    

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:4] = noise_scales.quat * noise_level
        noise_vec[4:7] = noise_scales.ang_vel * noise_level
        noise_vec[7] = noise_scales.dof_pos * noise_level
        noise_vec[8:10] = noise_scales.dof_vel * noise_level
        noise_vec[10:12] = 0. 
        noise_vec[12:15] = noise_scales.pos * noise_level
        noise_vec[15:17] = 0. # previous actions

        return noise_vec
    
    def _init_buffers(self):
        super()._init_buffers()
        self.base_pos = self.root_states[:, :3]
        self.rand_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.cfg.env.num_single_obs, dtype=torch.float, device=self.device))
        for _ in range(self.cfg.env.c_frame_stack):
            if self.cfg.terrain.measure_heights:
                self.critic_history.append(torch.zeros(
                self.num_envs, self.cfg.env.single_num_privileged_obs + self.cfg.terrain.num_height, dtype=torch.float, device=self.device))
            else:
                self.critic_history.append(torch.zeros(
                self.num_envs, self.cfg.env.single_num_privileged_obs, dtype=torch.float, device=self.device))

        self.Circle_Trajectory = []
        self.Real_Trajectory = deque(maxlen=self.cfg.env.trajectory_length)
        for i in range(self.cfg.env.trajectory_length):
            if i <=20:
                self.Circle_Trajectory.append([0.25*(i/10)*(i/10),0])
            else:
                self.Circle_Trajectory.append([1 + 5 * math.sin((i-20)/50) , 5 - 5 * math.cos((i-20)/50)])
        
        for _ in range(self.cfg.env.trajectory_length):
            self.Real_Trajectory.append(torch.zeros(
                self.num_envs, 3, dtype=torch.float, device=self.device))
        self.Tra_pos_now = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.Real_Trajectory_Point = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.Real_Trajectory_Point[:] = self.base_init_state[:3]

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            

    def _process_rigid_shape_props(self, props, env_id):
        # rigid shapes set rand friction and restitution
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
                
            self.env_frictions[env_id] = self.friction_coeffs[env_id]
        return props
    
    def _process_dof_props(self, props, env_id):
        props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        props["stiffness"].fill(0.0)
        props["damping"].fill(0.0)
        return props

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
                self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0
    
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
            if self.commands[env_ids[i], 1] >= yaw_limit[i]:
                self.commands[env_ids[i], 1] = yaw_limit[i]
            if self.commands[env_ids[i], 1] <= -yaw_limit[i]:
                self.commands[env_ids[i], 1] = -yaw_limit[i]

    #------------ reward functions----------------
    def _reward_tracking_trajectory(self):
        # Tracking of trajectory points
        trajectory_error = torch.sum(torch.square(self.Real_Trajectory_Point[:, :2] - self.Tra_pos_now[:]), dim = 1)
        return torch.exp(-trajectory_error/self.cfg.rewards.tracking_sigma)

    def _reward_balance(self):
        balance_error = torch.square(self.base_lin_vel[:, 1])
        return torch.exp(-balance_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 1]) + torch.square(self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 1] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos[:,1].unsqueeze(1) + 0.5236).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos[:,1].unsqueeze(1) - 0.5236).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
