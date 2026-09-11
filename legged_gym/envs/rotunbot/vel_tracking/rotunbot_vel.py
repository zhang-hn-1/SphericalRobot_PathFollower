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
from .rotunbot_vel_config import RotunbotVelCfg
from scipy.spatial.transform import Rotation

class PIDController:
    def __init__(self, kp, ki, kd, num_env, device=None):
        """
        初始化PID控制器
        参数:kp (float): 比例增益;ki (float): 积分增益;kd (float): 微分增益
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.last_error = torch.zeros(num_env, device=device, dtype=torch.float)
        self.integral = torch.zeros(num_env, device=device, dtype=torch.float)

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

def copysign_new(a, b):

    a = torch.tensor(a, device=b.device, dtype=torch.float)
    a = a.expand_as(b)
    return torch.abs(a) * torch.sign(b)

def get_euler_rpy(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[..., qw] * q[..., qx] + q[..., qy] * q[..., qz])
    cosr_cosp = q[..., qw] * q[..., qw] - q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] + q[..., qz] * q[..., qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[..., qw] * q[..., qy] - q[..., qz] * q[..., qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign_new(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[..., qw] * q[..., qz] + q[..., qx] * q[..., qy])
    cosy_cosp = q[..., qw] * q[..., qw] + q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] - q[..., qz] * q[..., qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_rpy(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=-1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

class RotunbotVel(LeggedRobot):
    '''
    Rotunbot is a class that represents a custom environment for a spherical robot.

    加入延迟和更多随机噪声
    随机延迟
    关节参数噪声
    质量、惯性矩阵、摩擦系数噪声

    '''
    cfg : RotunbotVelCfg

    def __init__(self, cfg:RotunbotVelCfg , sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
         # 添加历史状态存储
        self.last_base_lin_vel = torch.zeros_like(self.base_lin_vel)
        self.last_base_ang_vel = torch.zeros_like(self.base_ang_vel)
        self.last_lagged_base_lin_vel = torch.zeros_like(self.base_lin_vel)  # 复制当前线速度
        self.last_lagged_base_ang_vel = torch.zeros_like(self.base_ang_vel) # 复制当前角速度
        self.last_lagged_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_lagged_dof_vel = torch.zeros_like(self.dof_vel)
        # self.last_torques = torch.zeros((self.num_envs, self.num_actions), device=self.device)  # 存储上一时刻力矩
        self.step_counter = 0
        self.print_interval = 5  # 每50步打印一次
        self.data_print = True

        self.PID_FirstAxis = PIDController(35, 0, 0, self.num_envs, device=self.device)
        self.PID_SecondAxis = PIDController(300, 20, 150, self.num_envs, device=self.device) #PIDController(200, 20, 120,self.num_envs)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        #record last state
        self.step_counter += 1
    
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            if getattr(self, "contact_yaw_damping_enabled", False):
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                self.gym.refresh_net_contact_force_tensor(self.sim)
            self._apply_contact_yaw_damping()
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            #加入延迟
            #dof_lag
            if self.cfg.domain_rand.add_dof_lag:
                q = self.dof_pos
                dq = self.dof_vel
                self.dof_lag_buffer[:,:,1:] = self.dof_lag_buffer[:,:,:self.cfg.domain_rand.dof_lag_timesteps_range[1]].clone()
                self.dof_lag_buffer[:,:,0] = torch.cat((q, dq), 1).clone()
            if self.cfg.domain_rand.add_imu_lag:
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.base_quat[:] = self.root_states[:, 3:7]
                self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
                self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
                base_ret = Rotation.from_quat(self.base_quat.cpu().numpy())
                self.base_euler_tensor = torch.as_tensor(base_ret.as_euler('xyz'),dtype=torch.float,device=self.device)
                self.imu_lag_buffer[:,:,1:] = self.imu_lag_buffer[:,:,:self.cfg.domain_rand.imu_lag_timesteps_range[1]].clone()
                self.imu_lag_buffer[:,:,0] = torch.cat((self.base_lin_vel, self.base_ang_vel, self.base_euler_tensor ), 1).clone()
        
        self.post_physics_step()
        
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # Optional task-specific quantities derived from the center trajectory.
        # The clean velocity task defines this hook before rewards and
        # observations are computed; other tasks are unchanged.
        trajectory_update = getattr(self, "_update_trajectory_quantities", None)
        if trajectory_update is not None:
            trajectory_update()

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_base_lin_vel = self.base_lin_vel.clone()  # 复制当前线速度
        self.last_base_ang_vel = self.base_ang_vel.clone() # 复制当前角速度
        self.last_lagged_base_lin_vel = self.lagged_base_lin_vel.clone()  # 复制当前线速度
        self.last_lagged_base_ang_vel = self.lagged_base_ang_vel.clone() # 复制当前角速度
        self.last_lagged_dof_pos = self.lagged_dof_pos.clone()
        self.last_lagged_dof_vel = self.lagged_dof_vel.clone()

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def compute_observations(self):
        """ Computes observations
        """
         # 计算速度差 (当前速度 - 上一时刻速度)
        lin_vel_diff = self.base_lin_vel - self.last_base_lin_vel  # 线速度差
        ang_vel_diff = self.base_ang_vel - self.last_base_ang_vel # 角速度差
        quat = self.base_quat.cpu().numpy()
        # print(quat)

        # 检查零范数四元数
        # norms = np.linalg.norm(quat, axis=1)
        # print(quat)
        # if np.any(norms == 0):
        #     print("Warning: Found zero norm quaternions. Replacing with default.")
        #     quat[norms == 0] = [0, 0, 0, 1]  # 替换为单位四元数

        # 归一化四元数
        # quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)

        # 转换为旋转
        base_ret = Rotation.from_quat(quat)
        # base_ret = Rotation.from_quat(self.base_quat.cpu().numpy())
        self.base_euler_tensor = torch.as_tensor(base_ret.as_euler('xyz'),dtype=torch.float,device=self.device)
        privileged_obs_buf = torch.cat((
            self.commands,  # 2 
            self.base_euler_tensor * self.obs_scales.quat,  # 3
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.dof_pos  * self.obs_scales.dof_pos,  # 2
            self.dof_vel * self.obs_scales.dof_vel,  # 2
            self.actions,  # 2
            self.env_frictions,  # 1
            self.body_mass / 10.,  # 1 # sum of all fix link mass
            self.total_mass / 10.  # 1 # sum of all fix link mass
        ), dim=-1)

        if self.cfg.domain_rand.add_dof_lag:
            if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                self.dof_lag_timestep = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.dof_lag_timestep > self.last_dof_lag_timestep + 1
                self.dof_lag_timestep[cond] = self.last_dof_lag_timestep[cond] + 1
                self.last_dof_lag_timestep = self.dof_lag_timestep.clone()
            self.lagged_dof_pos = self.dof_lag_buffer[torch.arange(self.num_envs), :self.num_actions, self.dof_lag_timestep.long()]
            self.lagged_dof_vel = self.dof_lag_buffer[torch.arange(self.num_envs), -self.num_actions:, self.dof_lag_timestep.long()]  
        else:
            self.lagged_dof_pos = self.dof_pos
            self.lagged_dof_vel = self.dof_vel

        if self.cfg.domain_rand.add_imu_lag:    
            if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                self.imu_lag_timestep = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.imu_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.imu_lag_timestep > self.last_imu_lag_timestep + 1
                self.imu_lag_timestep[cond] = self.last_imu_lag_timestep[cond] + 1
                self.last_imu_lag_timestep = self.imu_lag_timestep.clone()
            self.lagged_imu = self.imu_lag_buffer[torch.arange(self.num_envs), :, self.imu_lag_timestep.long()]
            self.lagged_base_lin_vel = self.lagged_imu[:,:3].clone()
            self.lagged_base_ang_vel = self.lagged_imu[:,3:6].clone()
            self.lagged_base_euler_xyz = self.lagged_imu[:,-3:].clone()
        # no imu lag
        else:              
            self.lagged_base_lin_vel = self.base_lin_vel[:,:3]
            self.lagged_base_ang_vel = self.base_ang_vel[:,:3]
            self.lagged_base_euler_xyz = self.base_euler_tensor[:,-3:]
        '''
        没加入延迟
        self.obs_buf = torch.cat((  self.commands[:, :2],
                                    # self.base_quat,
                                    # self.root_states[:, 7:10],
                                    # self.root_states[:, 10:13],
                                    self.base_lin_vel, # 当前球坐标系线速度
                                    self.base_ang_vel, # 当前球坐标系角速度
                                    self.dof_pos[:,1].unsqueeze(1) ,
                                    self.dof_vel,
                                    self.actions,
                                    self.last_base_lin_vel,         # 上一时刻线速度
                                    self.last_base_ang_vel,         # 上一时刻角速度    
                                    self.last_actions,              
                                    self.last_dof_vel
                                    # self.last_torques,              # [2] 上一时刻力矩
                                    # self.last_dof_vel,
                                    ),dim=-1)
        加入延迟，加入电机状态
        self.obs_buf = torch.cat((  self.commands[:, :2],    # 2
                                    self.lagged_base_euler_xyz,  # 3
                                    # self.root_states[:, 7:10],
                                    # self.root_states[:, 10:13],
                                    self.lagged_base_lin_vel, # 3当前球坐标系线速度
                                    self.lagged_base_ang_vel, # 3当前球坐标系角速度
                                    self.lagged_dof_pos[:,1].unsqueeze(1) ,
                                    self.lagged_dof_vel,
                                    self.actions,
                                    self.last_lagged_base_lin_vel,         # 上一时刻线速度
                                    self.last_lagged_base_ang_vel,         # 上一时刻角速度    
                                    # lin_vel_diff,
                                    # ang_vel_diff,  
                                    self.last_actions,              
                                    self.last_lagged_dof_vel
                                    # self.last_torques,              # [2] 上一时刻力矩
                                    # self.last_dof_vel,
                                    ),dim=-1)
        '''
        
        self.obs_buf = torch.cat((  self.commands[:, :2]* self.obs_scales.command,    # 2
                                    self.lagged_base_euler_xyz*self.obs_scales.quat,  # 3
                                    # self.root_states[:, 7:10],
                                    # self.root_states[:, 10:13],
                                    self.lagged_base_lin_vel*self.obs_scales.lin_vel, # 3当前球坐标系线速度
                                    self.lagged_base_ang_vel*self.obs_scales.ang_vel, # 3当前球坐标系角速度
                                    self.lagged_dof_pos[:,1].unsqueeze(1)*self.obs_scales.dof_pos ,
                                    self.lagged_dof_vel[:,0].unsqueeze(1)*self.obs_scales.dof_vel,
                                    self.actions,             # 2
                                    # lin_vel_diff,
                                    # ang_vel_diff,  
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
        addition = actions * self.cfg.control.action_scale
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
            # actions_scaled[:,0] = torch.clip(actions[:,0] * self.cfg.control.first_actionScale, -6, 6)
            # actions_scaled[:,1] = torch.clip(actions[:,1] * self.cfg.control.second_actionScale, -0.5236, 0.5236)
            actions_scaled[:,0] = torch.clip(actions[:,0] * self.cfg.control.first_actionScale, -5, 5)
            actions_scaled[:,1] = torch.clip(actions[:,1] * self.cfg.control.second_actionScale, -0.45, 0.45)
            # actions_scaled[:,0] = 1
            # actions_scaled[:,1] = 0.3
            self.output_actions = actions_scaled
            
            # torques[:,0] =  25 * (actions_scaled[:,0] - self.dof_vel[:,0]) - 1 * (self.dof_vel[:,0] - self.last_dof_vel[:,0]) / self.dt
            torques[:,0] =  35 * (actions_scaled[:,0] - self.dof_vel[:,0])
            # torques[:,0] = self.PID_FirstAxis.compute(actions_scaled[:,0], self.dof_pos[:,1], self.sim_params.dt)
            # torques[:,1] = 15 * ( actions_scaled[:,1]  - self.dof_pos[:,1]) - 5 * self.dof_vel[:,1]
            torques[:,1] = 300 * ( actions_scaled[:,1]  - self.dof_pos[:,1]) - 150 * self.dof_vel[:,1]
            # torques[:,1] = self.PID_SecondAxis.compute(actions_scaled[:,1], self.dof_pos[:,1], self.sim_params.dt)
            torques[:,0] =  21.17 * (actions[:,0] - self.dof_vel[:,0]) - 0.97*(self.dof_vel[:,0] - self.last_dof_vel[:,0])/self.dt
            torques[:,1] =  297.46 * (actions[:,1]  - self.dof_pos[:,1]) - 149.97 * self.dof_vel[:,1]
            # torques[:,0] =  136.46 * (actions[:,0] - self.dof_vel[:,0]) - 162.73 *(self.dof_vel[:,0] - self.last_dof_vel[:,0])/self.dt
            # torques[:,1] =  299.98 * (actions[:,1]  - self.dof_pos[:,1]) - 98.82 * self.dof_vel[:,1]
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        torques[:,0] = torch.clip(torques[:,0], -self.cfg.control.torque_limits_1, self.cfg.control.torque_limits_1)
        torques[:,1] = torch.clip(torques[:,1], -self.cfg.control.torque_limits_2, self.cfg.control.torque_limits_2)
        

        if self.cfg.domain_rand.randomize_torque:
            torques *= self.torque_multi
        
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
        
        noise_vec[2:5] = noise_scales.quat * noise_level * self.obs_scales.quat
        noise_vec[5:8] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[8:11] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[11] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[13:15] = 0.

        # noise_vec[2:5] = noise_scales.quat * noise_level
        # noise_vec[5:8] = noise_scales.lin_vel * noise_level
        # noise_vec[8:11] = noise_scales.ang_vel * noise_level
        # noise_vec[11] = noise_scales.dof_pos * noise_level
        # noise_vec[12:14] = noise_scales.dof_vel * noise_level
        # noise_vec[14:16] = 0.

        if self.cfg.terrain.measure_heights:
            noise_vec[23:205] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec
    
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

        self.init_randomize_props()

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)

        self.body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.init_body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.total_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)

        self.randomize_rigid_body_props(torch.arange(self.num_envs, device=self.device))
        self.randomize_dof_props(torch.arange(self.num_envs, device=self.device))
        
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self._refresh_actor_dof_props(torch.arange(self.num_envs, device=self.device))
       
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])


    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                restitution_range = self.cfg.domain_rand.restitution_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                restitution_buckets = torch_rand_float(restitution_range[0], restitution_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
                self.restitution_coeffs = restitution_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
                props[s].restitution = self.restitution_coeffs[env_id]

            self.env_frictions[env_id] = self.friction_coeffs[env_id]

        return props

    def _process_dof_props(self, props, env_id):
        props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        props["stiffness"].fill(0.0)
        props["damping"].fill(0.0)
        return props

    def _process_rigid_body_props(self, props, env_id):
        # props[0].mass is sum mass of all fix link
        # len(props) is revolute num + 1
        # add rand payload on fix body
        # input_solution = [33.1026,20.035 ,65.155, 0.2775, 2.9869, 0.1004, 0.2522, 0.2629, 0.4522,
        # 2.9845, 2.9383, 0.2128, 1.3646, 1.6273, 2.9998, 0.9882]
        # for i in range(len(props)):
        #     props[i].mass = input_solution[i]  # m
        #     props[i].inertia.x.x = input_solution[i * 3 + 3]  # Ixx
        #     props[i].inertia.y.y = input_solution[i * 3 + 4]  # Iyy
        #     props[i].inertia.z.z = input_solution[i * 3 + 5]  # Izz
        
        if self.cfg.domain_rand.randomize_base_mass:
            self.init_body_mass[env_id] = props[0].mass
            props[0].mass += self.payload_masses[env_id]
        self.body_mass[env_id] = props[0].mass

        # rand all link mass and recalculate total mass
        if self.cfg.domain_rand.randomize_link_mass:
            for i in range(1, len(props)):
                props[i].mass *= self.link_masses[env_id, i-1]
        for i in range(1, len(props)):    
            self.total_mass[env_id] += props[i].mass

        # rand fix body com
        if self.cfg.domain_rand.randomize_com:
             props[0].com = gymapi.Vec3(self.com_displacements[env_id, 0], self.com_displacements[env_id, 1],
                                    self.com_displacements[env_id, 2])

        # rand link com
        if self.cfg.domain_rand.randomize_link_com:
            for i in range(1, len(props)):
                props[i].com = gymapi.Vec3(self.link_com_displacements[env_id, i-1, 0], self.link_com_displacements[env_id, i-1, 1],
                                           self.link_com_displacements[env_id, i-1, 2])     
        
        # rand fix body inertia
        if self.cfg.domain_rand.randomize_base_inertia:
            props[0].inertia.x.x *= self.base_inertia_x[env_id]
            props[0].inertia.y.y *= self.base_inertia_y[env_id]
            props[0].inertia.z.z *= self.base_inertia_z[env_id]
        
        # rand link inertia
        if self.cfg.domain_rand.randomize_link_inertia:
            for i in range(1, len(props)):
                props[i].inertia.x.x *= self.link_inertia_x[env_id, i-1]
                props[i].inertia.y.y *= self.link_inertia_y[env_id, i-1]
                props[i].inertia.z.z *= self.link_inertia_z[env_id, i-1]
                
        return props
    
    def _refresh_actor_dof_props(self, env_ids):
        ''' Refresh the dof properties of the actor in the given environments, i.e.
            dof friction, damping, armature
        '''
        for env_id in env_ids:
            dof_props = self.gym.get_actor_dof_properties(self.envs[env_id], 0)

            for i in range(self.num_dof):
                if self.cfg.domain_rand.randomize_joint_friction:
                    if self.cfg.domain_rand.randomize_joint_friction_each_joint:
                        dof_props["friction"][i] *= self.joint_friction_coeffs[env_id, i]
                    else:    
                        dof_props["friction"][i] *= self.joint_friction_coeffs[env_id, 0]
                
                if self.cfg.domain_rand.randomize_joint_armature:
                    if self.cfg.domain_rand.randomize_joint_armature_each_joint:
                        dof_props["armature"][i] = self.joint_armatures[env_id, i]
                    else:
                        dof_props["armature"][i] = self.joint_armatures[env_id, 0]
            self.gym.set_actor_dof_properties(self.envs[env_id], 0, dof_props)

    #------------ randomization functions----------------
    def randomize_rigid_body_props(self, env_ids):
        ''' Randomise some of the rigid body properties of the actor in the given environments, i.e.
            sample the mass, centre of mass position, friction and restitution.'''
        if self.cfg.domain_rand.randomize_base_mass:
            min_payload, max_payload = self.cfg.domain_rand.added_mass_range

            self.payload_masses[env_ids] = torch_rand_float(min_payload, max_payload, (len(env_ids), 1), device=self.device)
        
        if self.cfg.domain_rand.randomize_link_mass:
            min_link_mass, max_link_mass = self.cfg.domain_rand.added_link_mass_range

            self.link_masses[env_ids] = torch_rand_float(min_link_mass, max_link_mass, (len(env_ids), self.num_bodies-1), device=self.device)

        if self.cfg.domain_rand.randomize_com:
            comx_displacement, comy_displacement, comz_displacement = self.cfg.domain_rand.com_displacement_range
            self.com_displacements[env_ids, :] = torch.cat((torch_rand_float(comx_displacement[0], comx_displacement[1], (len(env_ids), 1), device=self.device),
                                                            torch_rand_float(comy_displacement[0], comy_displacement[1], (len(env_ids), 1), device=self.device),
                                                            torch_rand_float(comz_displacement[0], comz_displacement[1], (len(env_ids), 1), device=self.device)),
                                                            dim=-1)
        
        if self.cfg.domain_rand.randomize_link_com:
            comx_displacement, comy_displacement, comz_displacement = self.cfg.domain_rand.link_com_displacement_range
            self.link_com_displacements[env_ids, :, :] = torch.cat((torch_rand_float(comx_displacement[0], comx_displacement[1], (len(env_ids), self.num_bodies-1, 1), device=self.device),
                                                                    torch_rand_float(comy_displacement[0], comy_displacement[1], (len(env_ids), self.num_bodies-1, 1), device=self.device),
                                                                    torch_rand_float(comz_displacement[0], comz_displacement[1], (len(env_ids), self.num_bodies-1, 1), device=self.device)),
                                                                    dim=-1)
        if self.cfg.domain_rand.randomize_base_inertia:
            inertia_x, inertia_y, inertia_z = self.cfg.domain_rand.base_inertial_range
            self.base_inertia_x[env_ids, :, :] = torch_rand_float(inertia_x[0], inertia_x[1], (len(env_ids), 1), device=self.device)
            self.base_inertia_y[env_ids, :, :] = torch_rand_float(inertia_y[0], inertia_y[1], (len(env_ids), 1), device=self.device)
            self.base_inertia_z[env_ids, :, :] = torch_rand_float(inertia_z[0], inertia_z[1], (len(env_ids), 1), device=self.device)
            
        if self.cfg.domain_rand.randomize_link_inertia:
            inertia_x, inertia_y, inertia_z = self.cfg.domain_rand.link_inertial_range
            self.link_inertia_x[env_ids, :, :] = torch_rand_float(inertia_x[0], inertia_x[1], (len(env_ids), self.num_bodies-1), device=self.device)
            self.link_inertia_y[env_ids, :, :] = torch_rand_float(inertia_y[0], inertia_y[1], (len(env_ids), self.num_bodies-1), device=self.device)
            self.link_inertia_z[env_ids, :, :] = torch_rand_float(inertia_z[0], inertia_z[1], (len(env_ids), self.num_bodies-1), device=self.device)
    
    def randomize_dof_props(self, env_ids):
        # Randomise the motor strength:
        # rand ouput torque
        if self.cfg.domain_rand.randomize_torque:
            motor_strength_ranges = self.cfg.domain_rand.torque_multiplier_range
            self.torque_multi[env_ids] = torch_rand_float(motor_strength_ranges[0], motor_strength_ranges[1], (len(env_ids),self.num_actions), device=self.device)

        # rand joint friction set in sim
        if self.cfg.domain_rand.randomize_joint_friction:
            if self.cfg.domain_rand.randomize_joint_friction_each_joint:
                for i in range(self.num_dofs):
                    range_key = f'joint_{i+1}_friction_range'
                    friction_range = getattr(self.cfg.domain_rand, range_key)
                    self.joint_friction_coeffs[env_ids, i] = torch_rand_float(friction_range[0], friction_range[1], (len(env_ids), 1), device=self.device).reshape(-1)
            else:                      
                joint_friction_range = self.cfg.domain_rand.joint_friction_range
                self.joint_friction_coeffs[env_ids] = torch_rand_float(joint_friction_range[0], joint_friction_range[1], (len(env_ids), 1), device=self.device)
        
        if self.cfg.domain_rand.randomize_joint_armature:
            if self.cfg.domain_rand.randomize_joint_armature_each_joint:
                for i in range(self.num_dofs):
                    range_key = f'joint_{i+1}_armature_range'
                    armature_range = getattr(self.cfg.domain_rand, range_key)
                    self.joint_armatures[env_ids, i] = torch_rand_float(armature_range[0], armature_range[1], (len(env_ids), 1), device=self.device).reshape(-1)
            else:
                joint_armature_range = self.cfg.domain_rand.joint_armature_range
                self.joint_armatures[env_ids] = torch_rand_float(joint_armature_range[0], joint_armature_range[1], (len(env_ids), 1), device=self.device)
            
    def randomize_lag_props(self,env_ids): 
        """ random add lag
        """
                      
        if self.cfg.domain_rand.add_dof_lag:
            self.dof_lag_buffer[env_ids, :, :] = 0.0
            if self.cfg.domain_rand.randomize_dof_lag_timesteps:
                self.dof_lag_timestep[env_ids] = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0],
                                                        self.cfg.domain_rand.dof_lag_timesteps_range[1]+1, (len(env_ids),),device=self.device)
                if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                    self.last_dof_lag_timestep[env_ids] = self.cfg.domain_rand.dof_lag_timesteps_range[1]
            else:
                self.dof_lag_timestep[env_ids] = self.cfg.domain_rand.dof_lag_timesteps_range[1]
 
        if self.cfg.domain_rand.add_imu_lag:                
            self.imu_lag_buffer[env_ids, :, :] = 0.0   
            if self.cfg.domain_rand.randomize_imu_lag_timesteps:
                self.imu_lag_timestep[env_ids] = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0],
                                                        self.cfg.domain_rand.imu_lag_timesteps_range[1]+1, (len(env_ids),),device=self.device)
                if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                    self.last_imu_lag_timestep[env_ids] = self.cfg.domain_rand.imu_lag_timesteps_range[1]
            else:
                self.imu_lag_timestep[env_ids] = self.cfg.domain_rand.imu_lag_timesteps_range[1]
    
                      
    
    def init_randomize_props(self):
        ''' Initialize torch tensors for random properties
        '''
        if self.cfg.domain_rand.randomize_base_mass:
            self.payload_masses = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,requires_grad=False)
            
        if self.cfg.domain_rand.randomize_com:
            self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                        requires_grad=False)
            
        if self.cfg.domain_rand.randomize_base_inertia:
            self.base_inertia_x = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
            self.base_inertia_y = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
            self.base_inertia_z = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
            
        if self.cfg.domain_rand.randomize_link_mass:
            self.link_masses = torch.ones(self.num_envs, self.num_bodies-1, dtype=torch.float, device=self.device,requires_grad=False)
            
        if self.cfg.domain_rand.randomize_link_com:
            self.link_com_displacements = torch.zeros(self.num_envs, self.num_bodies-1, 3, dtype=torch.float, device=self.device, requires_grad=False)
            
        if self.cfg.domain_rand.randomize_link_inertia:
            self.link_inertia_x = torch.ones(self.num_envs, self.num_bodies-1, dtype=torch.float, device=self.device, requires_grad=False)
            self.link_inertia_y = torch.ones(self.num_envs, self.num_bodies-1, dtype=torch.float, device=self.device, requires_grad=False)
            self.link_inertia_z = torch.ones(self.num_envs, self.num_bodies-1, dtype=torch.float, device=self.device, requires_grad=False)
            
        if self.cfg.domain_rand.randomize_friction:
            self.friction = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,requires_grad=False)     
               
        if self.cfg.domain_rand.randomize_joint_friction_each_joint:
            self.joint_friction_coeffs = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,requires_grad=False)
        else:
            self.joint_friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device,requires_grad=False)
            
        if self.cfg.domain_rand.randomize_joint_armature_each_joint:
            self.joint_armatures = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,requires_grad=False)  
        else:
            self.joint_armatures = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,requires_grad=False)
          
        if self.cfg.domain_rand.randomize_torque:
            self.torque_multi = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,requires_grad=False)
            
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
        yaw_limit = torch.abs(self.commands[env_ids, 0]/1.4)
        for i in range(len(env_ids)):
            if self.commands[env_ids[i], 1] > yaw_limit[i]:
                self.commands[env_ids[i], 1] = yaw_limit[i]
            if self.commands[env_ids[i], 1] < -yaw_limit[i]:
                self.commands[env_ids[i], 1] = -yaw_limit[i]
        self.commands[env_ids, 0] = -0.4
        self.commands[env_ids, 1] = -0.2

    
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
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
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

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

        if self.cfg.domain_rand.add_dof_lag:
            self.dof_lag_buffer = torch.zeros(self.num_envs,self.num_actions * 2,self.cfg.domain_rand.dof_lag_timesteps_range[1]+1,device=self.device)
            if self.cfg.domain_rand.randomize_dof_lag_timesteps:
                self.dof_lag_timestep = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0],
                                                        self.cfg.domain_rand.dof_lag_timesteps_range[1]+1, (self.num_envs,),device=self.device)
                if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                    self.last_dof_lag_timestep = torch.ones(self.num_envs,device=self.device,dtype=int) * self.cfg.domain_rand.dof_lag_timesteps_range[1]
            else:
                self.dof_lag_timestep = torch.ones(self.num_envs,device=self.device) * self.cfg.domain_rand.dof_lag_timesteps_range[1]

        if self.cfg.domain_rand.add_imu_lag:
            self.imu_lag_buffer = torch.zeros(self.num_envs, 9, self.cfg.domain_rand.imu_lag_timesteps_range[1]+1,device=self.device)
            if self.cfg.domain_rand.randomize_imu_lag_timesteps:
                self.imu_lag_timestep = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0],
                                                        self.cfg.domain_rand.imu_lag_timesteps_range[1]+1, (self.num_envs,),device=self.device)
                if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                    self.last_imu_lag_timestep = torch.ones(self.num_envs,device=self.device,dtype=int) * self.cfg.domain_rand.imu_lag_timesteps_range[1]
            else:
                self.imu_lag_timestep = torch.ones(self.num_envs,device=self.device) * self.cfg.domain_rand.imu_lag_timesteps_range[1]
               
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        self.randomize_dof_props(env_ids)
        self._refresh_actor_dof_props(env_ids)
        self.randomize_lag_props(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

        self.last_base_lin_vel[env_ids] = 0
        self.last_base_ang_vel[env_ids] = 0

        self.PID_FirstAxis.reset(env_ids)
        self.PID_SecondAxis.reset(env_ids)

        self.reset_buf[env_ids] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    

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
            print(f"输出力矩: {self.torques[0, 0]:.4f}, {self.torques[0, 1]:.4f}")  
            print(f"输出: {self.output_actions[0, 0]:.4f}, {self.output_actions[0, 1]:.4f}")
            # print(f"输出: {self.output_actions[0, 0]:.4f}, {self.output_actions[0, 1]:.4f}")  
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
    
