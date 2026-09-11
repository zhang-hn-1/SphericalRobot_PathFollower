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
import math
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .rotunbot_target_obstacle_config import RotunbotTargetObstacleCfg
from scipy.spatial.transform import Rotation

class PIDController:
    def __init__(self, kp, ki, kd , num_env):
        """
        初始化PID控制器
        参数:kp (float): 比例增益;ki (float): 积分增益;kd (float): 微分增益
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.last_error = torch.zeros(num_env, device='cuda', dtype=torch.float)
        self.integral = torch.zeros(num_env, device='cuda', dtype=torch.float)

    def compute(self, setpoint , current_value, dt):
        
        error = setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        return output
    
    def reset(self , env_ids):
        self.last_error[env_ids] = 0.
        self.integral[env_ids] = 0.

class RotunbotTargetObstacle(LeggedRobot):
    '''
    Rotunbot is a class that represents a custom environment for a spherical robot.
    球形机器人端到端控制环境，给目标点位置，控制机器人到达目标点

    加入障碍物
    障碍物A:  位置:(1.5,1.5) 形状0.5×0.5正方形
    障碍物B:  位置:(-3.0,-3.0) 形状0.7×0.7正方形

    避免碰撞: 设定指令command 如果在障碍物范围内则指令为0
                            (判断指令在范围内，距离障碍物位置<0.717*边长+0.5)
    输入：

    奖励函数：
        到达目标的奖励
        lin_vel_y惩罚
        lin_vel_x限制
        ang_vel_yaw限制
        碰撞惩罚: 判断碰撞

    噪声：

    随机扰动：

    '''
    cfg : RotunbotTargetObstacleCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
         # 添加历史状态存储
        self.last_base_lin_vel = torch.zeros_like(self.base_lin_vel)
        self.last_base_ang_vel = torch.zeros_like(self.base_ang_vel)
        self.last_root_states = torch.zeros_like(self.root_states)
        # self.last_torques = torch.zeros((self.num_envs, self.num_actions), device=self.device)  # 存储上一时刻力矩
        self.step_counter = 0
        self.print_interval = 5  # 每50步打印一次
        self.data_print = True
        self.stop_distance = self.cfg.commands.stop_distance

        self.PID_FirstAxis = PIDController(35, 0, 0, self.num_envs)
        self.PID_SecondAxis = PIDController(200, 20, 120, self.num_envs)

    def compute_observations(self):
        """ Computes observations
        """
        # 37 个观测量
        base_ret = Rotation.from_quat(self.base_quat.cpu().numpy())
        self.base_euler_tensor = torch.as_tensor(base_ret.as_euler('xyz'),dtype=torch.float,device=self.device)
        # self.obs_buf = torch.cat((  self.commands[:, :2], # 2 目标位置点[x, y]  (设置目标朝向+2)
        #                             self.root_states[:, :3], # 3 当前地面坐标系位置
        #                             self.base_quat,       # 4 姿态四元数 （欧拉角）
        #                             self.root_states[:, 7:10], # 3 当前地面坐标系线速度
        #                             self.root_states[:, 10:13],# 3 当前地面坐标系角速度
        #                             self.dof_pos[:,1].unsqueeze(1) , # 1 当前关节角度
        #                             self.dof_vel,                    # 2 当前关节角速度
        #                             self.actions,                    # 2 上一时刻动作
        #                             self.last_root_states,           # 13 上一时刻状态
        #                             self.last_actions,               # 2 上一时刻动作
        #                             self.last_dof_vel                # 2 上一时刻关节角速度
        #                             ),dim=-1)
        #  19
        self.obs_buf = torch.cat((  self.commands[:, :2], # 2 目标位置点[x, y]  (设置目标朝向+2)
                                    self.root_states[:, :3], # 3 当前地面坐标系位置
                                    self.base_euler_tensor,       # 3 姿态四元数 （欧拉角）
                                    self.root_states[:, 7:10], # 3 当前地面坐标系线速度
                                    self.root_states[:, 10:13],# 3 当前地面坐标系角速度
                                    self.dof_pos[:,1].unsqueeze(1) , # 1 当前关节角度
                                    self.dof_vel,                    # 2 当前关节角速度
                                    self.actions,                    # 2 上一时刻动作
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
    
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)

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
        elif control_type=="R":
            torques = actions.clone().to(self.device)
            actions_scaled[:,0] = torch.clip(actions[:,0] * self.cfg.control.first_actionScale, -6, 6)  #(-8 , 8)
            actions_scaled[:,1] = torch.clip(actions[:,1] * self.cfg.control.second_actionScale, -0.5236, 0.5236)
            self.output_actions = actions_scaled
            
            # torques[:,0] =  25 * (actions_scaled[:,0] - self.dof_vel[:,0]) - 1 * (self.dof_vel[:,0] - self.last_dof_vel[:,0]) / self.dt
            # torques[:,0] =  35 * (actions_scaled[:,0] - self.dof_vel[:,0])
            torques[:,0] = self.PID_FirstAxis.compute(actions_scaled[:,0], self.dof_pos[:,1], self.sim_params.dt)
            # torques[:,1] = 15 * ( actions_scaled[:,1]  - self.dof_pos[:,1]) - 5 * self.dof_vel[:,1]
            # torques[:,1] = 200 * ( actions_scaled[:,1]  - self.dof_pos[:,1]) - 100 * self.dof_vel[:,1]
            torques[:,1] = self.PID_SecondAxis.compute(actions_scaled[:,1], self.dof_pos[:,1], self.sim_params.dt)
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
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:2] = 0. # commands
        
        # noise_vec[2:5] = noise_scales.pos * noise_level
        # noise_vec[5:9] = noise_scales.quat * noise_level
        # noise_vec[9:12] = noise_scales.lin_vel * noise_level
        # noise_vec[12:15] = noise_scales.ang_vel * noise_level
        # noise_vec[15] = noise_scales.dof_pos * noise_level
        # noise_vec[16:18] = noise_scales.dof_vel * noise_level
        # noise_vec[18:20] = 0. # previous actions
        # noise_vec[20:23] = noise_scales.pos * noise_level
        # noise_vec[23:27] = noise_scales.quat * noise_level
        # noise_vec[27:30] = noise_scales.lin_vel * noise_level
        # noise_vec[30:33] = noise_scales.ang_vel * noise_level
        # noise_vec[33:35] = 0.
        # noise_vec[35:37] = noise_scales.dof_vel * noise_level
        noise_vec[2:5] = noise_scales.pos * noise_level
        noise_vec[5:8] = noise_scales.quat * noise_level
        noise_vec[8:11] = noise_scales.lin_vel * noise_level
        noise_vec[11:14] = noise_scales.ang_vel * noise_level
        noise_vec[14] = noise_scales.dof_pos * noise_level
        noise_vec[15:17] = noise_scales.dof_vel * noise_level
        noise_vec[17:19] = 0. # previous actions

        if self.cfg.terrain.measure_heights:
            noise_vec[23:205] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec
    
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

    def _create_scene_assets(self):
        """Create assets shared by every environment instance.

        Scene-specific tasks can override this hook without duplicating the
        robot-loading and actor-index bookkeeping in ``_create_envs``.
        """
        obstacle_opts = gymapi.AssetOptions()
        obstacle_opts.fix_base_link = True

        return {
            "obstacle_A_asset": self.gym.create_box(
                self.sim, 0.4, 4.0, 1.2, obstacle_opts
            ),
            "obstacle_B_asset": self.gym.create_box(
                self.sim, 4.5, 0.4, 1.2, obstacle_opts
            ),
            "obstacle_A_color": gymapi.Vec3(0.6, 0.1, 0.0),
            "obstacle_B_color": gymapi.Vec3(0.0, 0.2, 0.6),
        }

    def _create_scene_actors(self, env_handle, env_id, scene_assets):
        """Create the fixed obstacles for one environment instance."""
        obstacle_A_pose = gymapi.Transform()
        obstacle_A_pose.p = gymapi.Vec3(3.5, 1.5, 0.6)
        obstacle_A_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        obstacle_B_pose = gymapi.Transform()
        obstacle_B_pose.p = gymapi.Vec3(-1.0, -3.0, 0.6)
        obstacle_B_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        obstacle_A_actor = self.gym.create_actor(
            env_handle,
            scene_assets["obstacle_A_asset"],
            obstacle_A_pose,
            "obstacleA",
            env_id,
            2,
            0,
        )
        obstacle_B_actor = self.gym.create_actor(
            env_handle,
            scene_assets["obstacle_B_asset"],
            obstacle_B_pose,
            "obstacleB",
            env_id,
            2,
            0,
        )
        self.gym.set_rigid_body_color(
            env_handle,
            obstacle_A_actor,
            0,
            gymapi.MESH_VISUAL,
            scene_assets["obstacle_A_color"],
        )
        self.gym.set_rigid_body_color(
            env_handle,
            obstacle_B_actor,
            0,
            gymapi.MESH_VISUAL,
            scene_assets["obstacle_B_color"],
        )

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        scene_assets = self._create_scene_assets()

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        robot_actor_indices = []
        self.envs = []
        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            # pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            robot_actor_indices.append(
                self.gym.get_actor_index(
                    env_handle, actor_handle, gymapi.DOMAIN_SIM
                )
            )
             
            self._create_scene_actors(env_handle, i, scene_assets)
            
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
        self.robot_actor_indices = torch.tensor(
            robot_actor_indices,
            dtype=torch.int32,
            device=self.device,
        )
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _process_dof_props(self, props, env_id):
        props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        props["stiffness"].fill(0.0)
        props["damping"].fill(0.0)
        return props

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["pos_x"][0], self.command_ranges["pos_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["pos_y"][0], self.command_ranges["pos_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 0] = 6
        self.commands[env_ids, 1] = 2
        if self.cfg.commands.command_yaw:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["yaw"][0], self.command_ranges["yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.5).unsqueeze(1)

        #判断指令是否在障碍物范围内
        # 障碍物A:  位置:(1.5,1.5) 形状0.5×0.5正方形 障碍物范围：x:[3.5-(0.4/2+0.45), 3.5+(0.4/2+0.45)] y:[1.5-(4/2+0.45), 1.5+(4/2+0.45)]
        # 障碍物B:  位置:(-3.0,-3.0) 形状0.7×0.7正方形 障碍物范围：x:[-1.0-(4.5/2+0.45), -1.0+(4.5/2+0.45)] y:[-2.0-(0.4/2+0.45), -2.0+(0.4/2+0.45)]
        for i in range(len(env_ids)):
            if self.commands[env_ids[i], 0] < 3.5-(0.4/2+0.45) and self.commands[env_ids[i], 0] > 3.5+(0.4/2+0.45):
                if self.commands[env_ids[i], 1] < 1.5-(4/2+0.45) and self.commands[env_ids[i], 1] > 1.5+(4/2+0.45):
                    self.commands[env_ids[i], 0] = 6.0
                    self.commands[env_ids[i], 1] = 1.0
            if self.commands[env_ids[i], 0] < -1.0-(4.5/2+0.45) and self.commands[env_ids[i], 0] > -1.0+(4.5/2+0.45):
                if self.commands[env_ids[i], 1] < -2.0-(0.4/2+0.45) and self.commands[env_ids[i], 1] > -2.0+(0.4/2+0.45):
                    self.commands[env_ids[i], 0] = -3.0
                    self.commands[env_ids[i], 1] = -3.5
        # self.commands[env_ids, :] = to_torch([8., 4.], device=self.device).repeat((len(env_ids), 1))
        
        # for i in range(len(env_ids)):
        #     if self.commands[env_ids[i], 0] <= 0.1 and self.commands[env_ids[i], 0] >= -0.1:
        #         self.commands[env_ids[i], 0] = 0
        # yaw_limit = torch.abs(self.commands[env_ids, 0]/4)
        # for i in range(len(env_ids)):
        #     if self.commands[env_ids[i], 1] > yaw_limit[i]:
        #         self.commands[env_ids[i], 1] = yaw_limit[i]
        #     if self.commands[env_ids[i], 1] < -yaw_limit[i]:
        #         self.commands[env_ids[i], 1] = -yaw_limit[i]

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        _actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.actor_root_state = gymtorch.wrap_tensor(_actor_root_state).view(self.num_envs, -1, 13)
        self.root_states = self.actor_root_state[:, 0, :]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # euler angles
        self.base_pos = self.root_states[:, :3]
        self.goal_dist = torch.zeros(self.num_envs, device=self.device)
        self.orientation_error = torch.zeros(self.num_envs, device=self.device)
        self.last_goal_dist = torch.zeros(self.num_envs, device=self.device)
        self.last_orientation_error = torch.zeros(self.num_envs, device=self.device)
        self.base_euler_tensor = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        
    def step(self, actions):
        self.step_counter += 1
        # self.last_base_lin_vel = self.root_states[:, 7:10].clone()  # 复制当前线速度
        # self.last_base_ang_vel = self.root_states[:, 10:13].clone() # 复制当前角速度
        # self.last_torques = self.torques.clone()  
        self.last_base_lin_vel = self.base_lin_vel.clone()  # 复制当前线速度
        self.last_base_ang_vel = self.base_ang_vel.clone() # 复制当前角速度
        self.last_root_states = self.root_states.clone()
        self.last_goal_dist = self.goal_dist.clone()
        self.last_orientation_error = self.orientation_error.clone()
        return super().step(actions)
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        self.goal_dist = torch.sqrt(torch.sum(torch.square(self.commands[:, :2] - self.root_states[:, :2]), dim=1))
        # if torch.abs(self.commands[:, 2] - self.base_euler_tensor[:, 2]) 
        self.orientation_error = torch.where(torch.abs(self.commands[:, 2] - self.base_euler_tensor[:, 2])>math.pi,
                                             2*math.pi-torch.abs(self.commands[:, 2] - self.base_euler_tensor[:, 2]),
                                             torch.abs(self.commands[:, 2] - self.base_euler_tensor[:, 2]))
        return super()._post_physics_step_callback()
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        # self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs

        self.arrived_target_buf = self.goal_dist < self.stop_distance 
        if self.cfg.commands.command_yaw:   
            self.arrived_orientation_buf = self.orientation_error < 0.1
            self.arrived_target_buf = self.arrived_target_buf & self.arrived_orientation_buf
        roll_cutoff = torch.abs(self.base_euler_tensor[:,0]) > 1.2
        pitch_cutoff = torch.abs(self.base_euler_tensor[:,1]) > 1.2
        x_cutoff = torch.abs(self.base_pos[:,0]) > 15.0
        y_cutoff = torch.abs(self.base_pos[:,1]) > 15.0
        robot_vel = torch.norm(self.base_lin_vel, dim=1)
        self.stop_buf = robot_vel < 0.1
        self.reset_buf |= ( self.stop_buf & self.arrived_target_buf )
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        self.reset_buf |= x_cutoff
        self.reset_buf |= y_cutoff

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = 0.
        self.dof_vel[env_ids] = 0.

        robot_actor_indices = self.robot_actor_indices[env_ids]
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(robot_actor_indices),
                                              len(robot_actor_indices))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.actor_root_state[env_ids, 0, :] = self.base_init_state
            self.actor_root_state[env_ids, 0, :3] += self.env_origins[env_ids]
        # base velocities
        # self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        robot_actor_indices = self.robot_actor_indices[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.actor_root_state),
                                                     gymtorch.unwrap_tensor(robot_actor_indices),
                                                     len(robot_actor_indices))
    
    def reset_idx(self, env_ids):
        # 重置时也要重置历史信息
        self.last_base_lin_vel[env_ids] = 0
        self.last_base_ang_vel[env_ids] = 0
        self.last_root_states[env_ids] = 0
        self.goal_dist[env_ids] = 0
        self.last_goal_dist[env_ids] = 0
        self.last_orientation_error[env_ids] = 0
        # self.last_torques[env_ids] = 0
        self.PID_FirstAxis.reset(env_ids)
        self.PID_SecondAxis.reset(env_ids)
        
        super().reset_idx(env_ids)

    #------------ reward functions----------------
    def _reward_close_to_target(self):
        # Reward for being close to the target
        goal_close = self.goal_dist - self.last_goal_dist
        # print(f"距离: {self.goal_dist[0]:.4f}")  # 只打印第一项
        # print(f"上次距离: {self.last_goal_dist[0]:.4f}")  # 只打印第一项
        # print('**********************')
        return -goal_close*self.cfg.rewards.close_para
    
    def _reward_close_to_orientation(self):
        # Reward for being close to the target
        
        orientation_close = self.orientation_error - self.last_orientation_error
        
        if self.cfg.commands.command_yaw: 
            return -orientation_close*self.cfg.rewards.close_para
        else:
            return 0
        
    def _reward_to_target(self):
        print(f"给定位置: ({self.commands[0, 0]:.4f},{self.commands[0, 1]:.4f},{self.commands[0, 2]:.4f})")  
        print(f"实际位置: ({self.root_states[0, 0]:.4f},{self.root_states[0, 1]:.4f},{self.base_euler_tensor[0, 2]:.4f})")  
        print('**********************')
        pos_error = torch.sum(torch.square(self.commands[:, :2] - self.root_states[:, :2]), dim=1)
        return torch.exp(-pos_error/self.cfg.rewards.tracking_sigma_main)

    def _reward_stop(self):
        # Reward for stopping
        robot_vel = torch.norm(self.base_lin_vel, dim=1)
        if self.cfg.commands.command_yaw:  
            return 50 * (self.goal_dist<self.stop_distance )* (self.orientation_error<0.08)
        else:
            return 50 * (self.goal_dist<self.stop_distance )
        
    def _reward_to_orientation(self):
        
        yaw_error = torch.square(self.orientation_error)
        if self.cfg.commands.command_yaw:
            return torch.exp(-yaw_error/self.cfg.rewards.tracking_sigma_yaw)
        else:
            return 0
    
    def _reward_lin_vel_limits(self):
        # Reward for staying within linear velocity limits
        lin_vel_limits = -(self.base_lin_vel[:,0] - self.command_ranges["lin_vel_x"][0]).clip(max=0.) # lower limit
        lin_vel_limits += (self.base_lin_vel[:,0] - self.command_ranges["lin_vel_x"][1]).clip(min=0.)
        return lin_vel_limits
        
    
    def _reward_ang_vel_limits(self):
        # Reward for staying within angular velocity limits
        ang_vel_limits = -(self.base_lin_vel[:,2] - self.command_ranges["ang_vel_yaw"][0]).clip(max=0.)
        ang_vel_limits += (self.base_lin_vel[:,2] - self.command_ranges["ang_vel_yaw"][1]).clip(min=0.)
        return ang_vel_limits

    def _reward_balance(self):
        lin_vel_y = torch.square(self.base_lin_vel[:, 1])
        return lin_vel_y

    def _reward_overturn(self):
        # Reward for staying upright
        roll = torch.abs(self.base_euler_tensor[:, 0]) > 0.5236
        pitch = torch.abs(self.base_euler_tensor[:, 1]) > 0.6 
        return (roll + pitch)
    
    def _reward_collision(self):
        # 障碍物A:  位置:(1.5,1.5) 形状0.5×0.5正方形 障碍物范围：x:[3.5-(0.4/2+0.45), 3.5+(0.4/2+0.45)] y:[1.5-(4/2+0.45), 1.5+(4/2+0.45)]
        # 障碍物B:  位置:(-3.0,-3.0) 形状0.7×0.7正方形 障碍物范围：x:[-1.0-(4.5/2+0.45), -1.0+(4.5/2+0.45)] y:[-2.0-(0.4/2+0.45), -2.0+(0.4/2+0.45)]
        in_A = ((self.base_pos[:, 0] > 3.5-(0.4/2+0.45)) & (self.base_pos[:, 0] < 3.5+(0.4/2+0.45))) & ((self.base_pos[:, 1] > 1.5-(4/2+0.45)) & (self.base_pos[:, 1] < 1.5+(4/2+0.45)))
        in_B = ((self.base_pos[:, 0] > -1.0-(4.5/2+0.45)) & (self.base_pos[:, 0] < -1.0+(4.5/2+0.45))) & ((self.base_pos[:, 1] > -2.0-(0.4/2+0.45)) & (self.base_pos[:, 1] < -2.0+(0.4/2+0.45)))
        return (in_A + in_B)
