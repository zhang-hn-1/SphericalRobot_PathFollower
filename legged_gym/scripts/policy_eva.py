'''
仿真策略执行文件
计算策略的成功率、平稳度、

'''
import math
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from isaacgym.terrain_utils import *

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import os
import csv
from legged_gym import LEGGED_GYM_ROOT_DIR

import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import time

import cma

    

class Policy_eva():

    def __init__(self):
        self._init_parameter()
        self._init_env(headless=True)
        self._init_policy()
        self._init_buffers()
    
    def _init_parameter(self):

        self.num_envs = 512
        self.num_input = 4 # 16 # 25
        self.num_obs = 20 # 16 # 25
        self.num_commands = 2

        # 初始化运行时间
        episode_length_s = 30.0 #9.0

        # 初始化终止条件参数
        self.stop_distance = 0.3
        self.stop_vel = 0.1

        self.dt = 0.02
        self.max_episode_length = np.ceil(episode_length_s / self.dt)
        self.episode_length_buf = 0

        # 初始化控制参数
        self.set_a_rate_limit = True
        self.first_vel_limits = 1.5
        self.second_pos_limits = 0.45
        self.rate_limit_1 = 0.02
        self.rate_limit_2 = 0.04
        self.clip_obs = 100
        self.clip_actions = 100
        self.x_ranges = [-5.0, 5.0]
        self.y_ranges = [-5.0, 5.0]
        
        # 初始化数据记录
        self.file_path = "data/state_data_raw_v.txt"
        self.sim_data = []
        self.sim_data_all = [[] for _ in range(self.num_envs)]
        self.raw_data = self.read_and_process_state_data(self.file_path, episode_length_s)

    def _init_policy(self):
        self.policy=torch.jit.load("legged_gym/scripts/policies/policy_nav_9_3.pt").to(self.device)
        print(self.policy)
        
    def read_raw_data(self, file_path):
        # 读取原始数据
        data = []
        with open(file_path, "r") as file:
            lines = file.readlines()
            # 跳过表头（第一行）
            header = lines[0].strip().split()
            for line in lines[1:]:
                # 跳过空行或注释行
                if line.strip() == "" or line.startswith("//"):
                    continue
                # 解析数据行并将值转换为 float
                values = [float(value) for value in line.strip().split()]
                data.append(dict(zip(header, values)))

        if data:
            # 将时间从 0 开始，并保留两位小数
            initial_time = data[0]["time"]
            initial_yaw = data[0]["yaw"]
            for row in data:
                row["time"] = round(row["time"] - initial_time, 2)
                row["yaw"] = row["yaw"] - initial_yaw

            # 提取时间序列
            times = [row["time"] for row in data]
            final_time = times[-1]

            # 生成完整的时间序列（间隔为 0.02）
            full_times = np.arange(0, final_time + 0.02, 0.02)

            # 创建插值后的数据列表
            interpolated_data = []
            for t in full_times:
                if t in times:
                    # 如果时间点存在，直接使用原始数据
                    row = next(row for row in data if row["time"] == t)
                    interpolated_data.append(row)
                else:
                    # 如果时间点缺失，进行插值
                    prev_row = next(row for row in reversed(data) if row["time"] < t)
                    next_row = next(row for row in data if row["time"] > t)
                    
                    # 插值每个字段
                    interpolated_row = {"time": round(t, 2)}
                    for key in prev_row.keys():
                        if key != "time":
                            prev_value = prev_row[key]
                            next_value = next_row[key]
                            interpolated_row[key] = prev_value + (next_value - prev_value) * ((t - prev_row["time"]) / (next_row["time"] - prev_row["time"]))
                    interpolated_data.append(interpolated_row)

        return interpolated_data[:600]
        
    def read_and_process_state_data(self, file_path, total_time, time_interval=0.02):
        """
        读取 state_data_raw_i.txt 文件，并处理时间项从 0 开始，保留两位小数。
        每隔 0.02s 取一行数据。
        参数:
            file_path (str): 文件路径。
        返回:
            list: 处理后的数据列表。
        """
        data = []
        with open(file_path, "r") as file:
            lines = file.readlines()
            # 跳过表头（第一行）
            header = lines[0].strip().split()
            for line in lines[1:]:
                # 跳过空行或注释行
                if line.strip() == "" or line.startswith("//"):
                    continue
                # 解析数据行并将值转换为 float
                values = [float(value) for value in line.strip().split()]
                data.append(dict(zip(header, values)))
        # 确保时间项从 0 开始
        initial_time = data[0]["time"]
        initial_yaw = data[0]["yaw"]
        for row in data:
            row["time"] = round(row["time"] - initial_time, 2)
            row["yaw"] =  (row["yaw"] - initial_yaw + math.pi) % (2 * math.pi) - math.pi
            row["px"] = row["px"] - 5.0
            row["py"] = -row["py"] + 5.0
            row["second_pos"] = -row["second_pos"]


        # 以 0.02s 的时间间隔取数据
        processed_data = []
        full_times = np.arange(0, total_time, time_interval)
        for t in full_times:
            # 找到最接近的时间点
            closest_row = min(data, key=lambda row: abs(row["time"] - t))
            processed_data.append(closest_row)

        return processed_data

    def _init_env(self,headless=True):
        # 初始化gym
        # self.num_envs = 1
        self.num_actions = 2
        self.gym = gymapi.acquire_gym()
        args = gymutil.parse_arguments(description="Spherical Robot control Example")
        self.headless = headless

        self.device = args.sim_device if args.use_gpu_pipeline else 'cpu'
        # 设置gym参数
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.substeps = 2
        sim_params.dt = self.dt
        sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        # set PhysX-specific parameters
        sim_params.physx.use_gpu = True
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0

        
        self.graphics_device_id = args.graphics_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        self.sim = self.gym.create_sim(args.compute_device_id, self.graphics_device_id, args.physics_engine, sim_params)

        self.viewer = None
        if self.headless == False:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        print(self.viewer)
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0
        self.gym.add_ground(self.sim, plane_params)
        
        asset_path = "{LEGGED_GYM_ROOT_DIR}/resources/robots/Rotunbot/urdf/Rotunbot_test2.urdf".format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # 机器人模型参数
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        #asset_options.self_collision = True
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = True
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        ball_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(ball_asset)

        initial_euler = [0.0, 0.0, 0.0]  # 初始欧拉角
        initial_euler_ = Rotation.from_euler('xyz', initial_euler)
        self.initial_quat = initial_euler_.as_quat()  # -> [w,x,y,z]

        initial_pose = gymapi.Transform()
        initial_pose.p = gymapi.Vec3( 0.0, 0.0, 0.4)
        initial_pose.r = gymapi.Quat( 0.0, 0.0, 0.0, 1.0)

        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.actor_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
                
            actor_handle = self.gym.create_actor(env_handle, ball_asset, initial_pose, "Rotunbot", i, 1, 0)

            props = self.gym.get_actor_dof_properties(env_handle, actor_handle)
            props["driveMode"] = (gymapi.DOF_MODE_EFFORT)
            props["stiffness"] = (0.0)
            props["damping"] = (0.0)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, props)
            
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
        
        self.gym.prepare_sim(self.sim)

    def _get_observations(self):
        # 获取观测值
        self.obs_buf = torch.cat((  self.commands[:, :2], # 2 目标位置点[x, y]  (设置目标朝向+2)
                                    self.root_states[:, :2], # 2 当前地面坐标系位置
                                    self.base_euler_tensor,       # 3 姿态四元数 （欧拉角）
                                    self.root_states[:, 7:10], # 3 当前地面坐标系线速度
                                    self.root_states[:, 10:13],# 3 当前地面坐标系角速度
                                    self.dof_pos[:,1].unsqueeze(1) , # 1 当前关节角度
                                    self.dof_vel,                    # 2 当前关节角速度
                                    self.last_root_states[:, :2],           # 2 上一时刻状态位置
                                    self.actions,                    # 2 上一时刻动作
                                    ),dim=-1)
        self.obs_buf = torch.clip(self.obs_buf, -self.clip_obs, self.clip_obs)

    def step(self):
        # 执行一步仿真
        # self.actions = action
        torch.cuda.empty_cache()
        
        actions =  self.policy(self.obs_buf.detach())
        self.actions = torch.clip(actions, -self.clip_actions, self.clip_actions).to(self.device)
        self.torques = self._compute_torques(self.actions).view(self.torques.shape)
        # print("-------------")
        # print(self.commands)
        # print(self.base_pos)
        # print(action)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        # dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # self.root_states = gymtorch.wrap_tensor(actor_root_state)
        # self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_pos[:] = self.root_states[:, :3]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        
        base_ret = Rotation.from_quat(self.base_quat.cpu().numpy())
        self.base_euler_tensor = torch.as_tensor(base_ret.as_euler('xyz'),dtype=torch.float,device=self.device)
        self.goal_dist = torch.sqrt(torch.sum(torch.square(self.commands[:, :2] - self.root_states[:, :2]), dim=1))

        
        for i in range(self.num_envs):
            self.sim_data_all[i].append({
                'time': self.episode_length_buf * self.dt,
                # 'px': self.root_states[i, 0].item(),
                # 'py': self.root_states[i, 1].item(),
                # 'roll': self.base_euler_tensor[i, 0].item(),
                # 'pitch': self.base_euler_tensor[i, 1].item(),
                # 'yaw': self.base_euler_tensor[i, 2].item(),
                # 'vx': self.base_lin_vel[i, 0].item(),
                # 'vy': self.base_lin_vel[i, 1].item(),
                # 'vz': self.base_lin_vel[i, 2].item(),
                # 'wx': self.base_ang_vel[i, 0].item(),
                # 'wy': self.base_ang_vel[i, 1].item(),
                # 'wz': self.base_ang_vel[i, 2].item(),
                'second_pos': self.dof_pos[i, 1].item(),
                'first_vel': self.dof_vel[i, 0].item()
            })
        self.check_termination()
        self._get_observations()
        self.episode_length_buf += 1
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_actions[:] = self.actions[:]
        self.last_root_states = self.root_states.clone()
        self.last_output_actions = self.output_actions.clone()

        # 画面渲染
        if self.viewer:
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

    def _compute_torques(self, actions):
        # 计算扭矩
        torques = actions.clone().to(self.device)
        actions_scaled = actions.clone().to(self.device)

        actions_scaled[:,0] = torch.clip(actions[:,0] * 1.0, -self.first_vel_limits, self.first_vel_limits)  #(-8 , 8)
        actions_scaled[:,1] = torch.clip(actions[:,1] * 0.5, -self.second_pos_limits, self.second_pos_limits)
        if self.set_a_rate_limit:
            actions_scaled[:,0] = torch.where(actions_scaled[:,0]- self.last_output_actions[:,0] > self.rate_limit_1,
                                                self.last_output_actions[:,0] + self.rate_limit_1,actions_scaled[:,0])
            actions_scaled[:,0] = torch.where(actions_scaled[:,0]- self.last_output_actions[:,0] < -self.rate_limit_1,
                                                self.last_output_actions[:,0] - self.rate_limit_1,actions_scaled[:,0])
            actions_scaled[:,1] = torch.where(actions_scaled[:,1]- self.last_output_actions[:,1] > self.rate_limit_2,
                                                self.last_output_actions[:,1] + self.rate_limit_2,actions_scaled[:,1])
            actions_scaled[:,1] = torch.where(actions_scaled[:,1]- self.last_output_actions[:,1] < -self.rate_limit_2,
                                                self.last_output_actions[:,1] - self.rate_limit_2,actions_scaled[:,1])
            
        self.output_actions = actions_scaled
        
       
        torques[:,0] =  21.17 * (actions_scaled[:,0] - self.dof_vel[:,0]) - 0.97*(self.dof_vel[:,0] - self.last_dof_vel[:,0])/self.dt
        torques[:,1] =  297.46 * (actions_scaled[:,1]  - self.dof_pos[:,1]) - 149.97 * self.dof_vel[:,1]
        # torques[:,1] = self.PID_SecondAxis.compute(actions_scaled[:,1], self.dof_pos[:,1], self.sim_params.dt)
        
        return torques
   
    def _init_buffers(self):
        # 初始化缓冲区
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.initial_root_states = torch.zeros_like(self.root_states)
        self.initial_root_states[:,3] = self.initial_quat[0]
        self.initial_root_states[:,4] = self.initial_quat[1]
        self.initial_root_states[:,5] = self.initial_quat[2]
        self.initial_root_states[:,6] = self.initial_quat[3]
        self.initial_root_states[:,2] = 0.4
        self.initial_dof_states = torch.zeros_like(self.dof_state)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        
        self.base_pos = self.root_states[:, :3]
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        self.output_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.last_output_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.commands = torch.zeros(self.num_envs, self.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_root_states = torch.zeros_like(self.root_states)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_env = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.arrived_target_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_dist = torch.zeros(self.num_envs, device=self.device)
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.distance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.minimize_distance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    def _reset_envs(self):
        # 重置环境
        self.episode_length_buf = 0
        self.sim_data.clear()
        self.sim_data_all = [[] for _ in range(self.num_envs)]
        self._reset_dofs()
        self._reset_root_states()

        self.last_actions[:] = 0.
        self.last_dof_vel[:] = 0.
    
    def _reset_dofs(self):
        # 重置关节状态
        
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.initial_dof_states))

    def _reset_root_states(self):
        # 重置根状态
        self.gym.set_actor_root_state_tensor(self.sim,gymtorch.unwrap_tensor(self.initial_root_states))

    def Policy_run(self):
        # 主体函数
        self._resample_commands()
        for i in range(int(rotunbot.max_episode_length)):
            self.step()
        success_rate = self.calculate_success_rate()
        print(f"success rate: {success_rate * 100:.2f}%")
    
    def _resample_commands(self):
        """ Randommly select commands of some environments

        """
        x_rand_vals = torch.rand((self.num_envs, 1), device=self.device)  # 生成 [0, 1) 的随机数
        x_interval_choice = torch.rand((self.num_envs, 1), device=self.device)  # 决定选择哪个区间
        x_rand_vals = torch.where(
            x_interval_choice < 0.5,  # 50% 概率选择第一个区间
            -5 + (x_rand_vals * 4.5),  # 映射到 [-5, -0.5]
            0.5 + (x_rand_vals * 4.5)  # 映射到 [0.5, 5]
        )
        y_rand_vals = torch.rand((self.num_envs, 1), device=self.device)  # 生成 [0, 1) 的随机数
        y_interval_choice = torch.rand((self.num_envs, 1), device=self.device)  # 决定选择哪个区间
        y_rand_vals = torch.where(
            y_interval_choice < 0.5,  # 50% 概率选择第一个区间
            -5 + (y_rand_vals * 4.5),  # 映射到 [-5, -0.5]
            0.5 + (y_rand_vals * 4.5)  # 映射到 [0.5, 5]
        )

        # 将生成的随机数赋值到 commands 的第一列
        self.commands[:, 0] = x_rand_vals.squeeze(1)
        self.commands[:, 1] = y_rand_vals.squeeze(1)

        self.minimize_distance = torch.norm(self.commands[:, :2], dim=1)
    
        # self.commands[:, 0] = -2.0
        # self.commands[:, 1] = 5.0

    def check_termination(self):
        """ Check if environments need to be reset
        """
        # self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
       
        self.arrived_target_buf = self.goal_dist < self.stop_distance 
        
        robot_vel = torch.norm(self.base_lin_vel, dim=1)
        self.stop_buf = robot_vel < self.stop_vel
        self.success_buf = self.arrived_target_buf 
        self.success_env = torch.where(
            self.success_buf.bool(),  # 如果 arrived_target_buf 为 True
            torch.ones_like(self.success_env),  # 将 success_buf 置为 1
            self.success_env  # 否则保持原值不变
        )
        move_distance = torch.norm(self.root_states[:, :2] - self.last_root_states[:, :2], dim=1)
        self.distance = torch.where(
            ~self.success_env.bool(),  # 如果 arrived_target_buf 为 True
            self.distance + move_distance,  # 将 success_buf 置为 1
            self.distance  # 否则保持原值不变
        )

    def calculate_success_rate(self):
        """
        根据 success_env 计算成功率。
        success_env 中的值为 1 表示成功，0 表示未成功。
        """
        # 统计成功的环境数量
        num_success = torch.sum(self.success_env).item()

        # 总环境数量
        num_envs = self.success_env.shape[0]

        # 计算成功率
        success_rate = num_success / num_envs if num_envs > 0 else 0.0

        # print(self.minimize_distance[0].item())
        # print(self.distance[0].item())

        spl = torch.sum(self.success_env* self.minimize_distance / (self.distance + 1e-6)).item() / num_envs

        print(f"成功环境数量: {num_success}/{num_envs}")
        print(f"成功率: {success_rate * 100:.2f}%")
        print(f"SPL: {spl * 100:.4f}%")

        return success_rate

        

if __name__ == '__main__':
    rotunbot = Policy_eva()
    rotunbot.Policy_run()

    