'''
仿真参数辨识的文件
实际机器人运行的数据:
优化参数三个关节的惯性矩阵的对角线值,即Ixx,Iyy,Izz 共9个变量
调节两个电机的参数Kp,Kd 共4个变量
使得轨迹误差最小
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
    

class SYSID():

    def __init__(self):
        self._init_parameter()
        self._init_env(headless=True)
        self._init_buffers()
        self._reset_envs(self.init_solution)
    
    def _init_parameter(self):

        self.num_envs = 2048
        self.num_input = 4 # 16 # 25

        # 初始化运行时间
        episode_length_s = 12.0 #9.0

        self.dt = 0.02
        self.max_episode_length = np.ceil(episode_length_s / self.dt)
        self.episode_length_buf = 0
        # 初始化参数 [m0, m1, m2, Ixx0, Iyy0, Izz0, Ixx1, Iyy1, Izz1, Ixx2, Iyy2, Izz2, Kp1, Kd1, Kp2, Kd2]
        #                                                                              12   13   14   15
        self.init_solution = [[0.35, 0.0, 2.0, 1.0]]
        # self.init_solution = [[30.225, 27.834, 73.4, 1.1392, 1.097461, 1.598825, 2.3603, 1.8803, 2.3606, 0.774154, 
        #                        1.782475, 1.490523, 0.35, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.27]]
        self.kPID = [[35.0, 0.0, 200.0, 100.0] for _ in range(self.num_envs)]
        # 初始化搜索范围
        # self.search_range = [
        #     [0.5, 3.0],  # Ixx0
        #     [0.5, 3.0],  # Iyy0
        #     [0.5, 3.0],  # Izz0
        #     [0.5, 3.0],  # Ixx1
        #     [0.5, 3.0],  # Iyy1
        #     [0.5, 3.0],  # Izz1
        #     [0.5, 3.0],  # Ixx2
        #     [0.5, 3.0],  # Iyy2
        #     [0.5, 3.0],  # Izz2
        #     [10.0, 100.0], # Kp1
        #     [0.0, 100.0],# Kd1
        #     [10.0, 300.0],# Kp2
        #     [10.0, 200.0] # Kd2
        # ]
        # self.search_range = [
        #     [20.0, 15.0, 65.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.35],  # 初始值
        #     [40.0, 35.0, 85.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 3.0, 3.0,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.35]  # 截止值
        # ]
        self.search_range = [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 4.0, 4.0]
        ]

        # 初始化数据记录
        self.file_path = "data/state_data_raw_v.txt"
        self.sim_data = []
        self.sim_data_all = [[] for _ in range(self.num_envs)]
        self.raw_data = self.read_and_process_state_data(self.file_path, episode_length_s)

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
        
        asset_path = "{LEGGED_GYM_ROOT_DIR}/resources/robots/Rotunbot/urdf/Rotunbot.urdf".format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
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

        initial_euler = [0.0, -0.11, 0.0]  # 初始欧拉角
        initial_euler_ = Rotation.from_euler('xyz', initial_euler)
        self.initial_quat = initial_euler_.as_quat()  # -> [w,x,y,z]

        initial_pose = gymapi.Transform()
        initial_pose.p = gymapi.Vec3( 0.0, 0.0, 0.4)
        initial_pose.r = gymapi.Quat( self.initial_quat[0], self.initial_quat[1], self.initial_quat[2], self.initial_quat[3])

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
            
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._init_rigid_body_props(body_props)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
        
        self.gym.prepare_sim(self.sim)
        
    def _process_rigid_body_props(self, props, input_solution, env_id):
        
        solution_length = len(input_solution)
        repeated_index = env_id % solution_length  # 循环分配参数

        # 获取当前环境的参数
        env_solution = input_solution[repeated_index]

        # 更新刚体属性
        for i in range(len(props)):
            props[i].mass = env_solution[i]  # m
            props[i].inertia.x.x = env_solution[i * 3 + 3]  # Ixx
            props[i].inertia.y.y = env_solution[i * 3 + 4]  # Iyy
            props[i].inertia.z.z = env_solution[i * 3 + 5]  # Izz
            props[i].com = gymapi.Vec3(env_solution[i * 3 + 16], env_solution[i * 3 + 17],
                                           env_solution[i * 3 + 18]) 
        return props
    
    def _init_rigid_body_props(self, props):
        if self.num_input == 4:
            return props

        for i in range(len(props)):
            props[i].mass = self.init_solution[0][i]
            props[i].inertia.x.x = self.init_solution[0][ i*3 + 3]  # Ixx
            props[i].inertia.y.y = self.init_solution[0][ i*3 + 4]
            props[i].inertia.z.z = self.init_solution[0][ i*3 + 5]
            props[i].com = gymapi.Vec3(self.init_solution[0][i * 3 + 16], self.init_solution[0][i * 3 + 17],
                                           self.init_solution[0][i * 3 + 18]) 
        return props
    
    def step(self):
        # 执行一步仿真
        # self.actions = action
        action =  self.get_action_sequence1(self.episode_length_buf)
        self.actions[:,0] = action[0]
        self.actions[:,1] = action[1]
        self.torques = self._compute_torques(self.actions,self.kPID).view(self.torques.shape)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        base_ret = Rotation.from_quat(self.base_quat.cpu().numpy())
        self.base_euler_tensor = torch.as_tensor(base_ret.as_euler('xyz'),dtype=torch.float,device=self.device)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
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
        self.episode_length_buf += 1
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_actions[:] = self.actions[:]

        # 画面渲染
        if self.viewer:
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

    def get_action_sequence1(self, episode_length_buf):
        """
        根据时间计数生成动作序列。
        参数:
            episode_length_buf (int): 时间计数，每 0.02 秒记一次。
        返回:
            float: 当前时间点的动作值。
        """
        # 将时间计数转换为秒
        time = episode_length_buf * self.dt  # self.dt = 0.02
        # 主轴动作序列
        if time < 1.0:
            action1 = 0.0
        elif 1.0 <= time < 2.0:
            # 第 2 秒到第 3 秒，线性上升
            action1 = 0.0 + (2.0 / 1.0) * (time - 1.0)
        elif 2.0 <= time < 5.0:
            # 其他时间，动作为 0
            action1 = 2.0
        elif 5.0 <= time < 8.0:
            action1 = -1.0
        elif 8.0 <= time < 9.0:
            action1 = 3.0
        elif 9.0 <= time < 10.0:
            action1 = 3.0 - (3.0 / 1.0) * (time - 9.0)  # 线性下降
        else:
            action1 = 0.0

        # 动作序列逻辑
        if time < 2.0:
            action2 = 0.0
        elif 2.0 <= time < 3.0:
            action2 = 0.0 + (0.4 / 2.0) * (time - 3.0)
        elif 3.0 <= time < 5.0:
            action2 = 0.4
        elif 5.0 <= time < 7.0:
            action2 = -0.3
        elif 7.0 <= time < 9.0:
            action2 = -0.3 + (0.75 / 2.0) * (time - 7.0)  # 线性上升
        elif 9.0 <= time < 10.0:
            action2 = 0.45 - (0.45 / 1.0) * (time - 9.0)  # 余弦信号
        else:
            action2 = 0.0

        # 生成第二个动作
        
        return [action1,action2]
    
    def get_action_sequence2(self, episode_length_buf):
        """
        根据时间计数生成动作序列。
        参数:
            episode_length_buf (int): 时间计数，每 0.02 秒记一次。
        返回:
            float: 当前时间点的动作值。
        """
        # 将时间计数转换为秒
        time = episode_length_buf * self.dt  # self.dt = 0.02
        # 主轴动作序列
        if time < 0.5:
            action1 = 0.0
        elif 0.0 <= time < 7.5:
            action1 = -3.0
        else:
            action1 = 0.0

        # 动作序列逻辑
        if time < 1.5:
            action2 = 0.0
        elif 1.5 <= time < 7.5:
            action2 = 0.4 * np.sin(2 * np.pi * (time - 1.5) / 6.0)
        else:
            action2 = 0.0

        # 生成第二个动作
        
        return [action1,action2]
    
    def _compute_torques(self, actions, kPID):
        # 计算扭矩
        torques = actions.clone().to(self.device)

        for i in range(self.num_envs):
            Kp1, Kd1, Kp2, Kd2 = kPID[i]

            torques[i, 0] = Kp1 * (actions[i, 0] - self.dof_vel[i, 0]) - Kd1 * (self.dof_vel[i, 0] - self.last_dof_vel[i, 0]) / self.dt
            torques[i, 1] = Kp2 * (actions[i, 1] - self.dof_pos[i, 1]) - Kd2 * self.dof_vel[i, 1]

        torques[:,0] = torch.clip(torques[:,0], -100, 100)
        torques[:,1] = torch.clip(torques[:,1], -100, 100)

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
        
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

    def _reset_envs(self, input_solution):
        # 重置环境
        self.episode_length_buf = 0
        self.sim_data.clear()
        self.sim_data_all = [[] for _ in range(self.num_envs)]
        self._reset_dofs()
        self._reset_root_states()
        if self.num_input != 4:
            self.reset_rigid_props(input_solution)
        self.reset_kPID(input_solution)

        self.last_actions[:] = 0.
        self.last_dof_vel[:] = 0.
    
    def _reset_dofs(self):
        # 重置关节状态
        
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.initial_dof_states))

    def _reset_root_states(self):
        # 重置根状态
        self.gym.set_actor_root_state_tensor(self.sim,gymtorch.unwrap_tensor(self.initial_root_states))

    def reset_rigid_props(self, input_solution):
        # 重置刚体属性
        for i, env in enumerate(self.envs):
            actor_handle = self.actor_handles[i]
            body_props = self.gym.get_actor_rigid_body_properties(env, actor_handle)
            body_props = self._process_rigid_body_props(body_props, input_solution, i)
            self.gym.set_actor_rigid_body_properties(env, actor_handle, body_props, recomputeInertia=True)
    
    def reset_kPID(self, input_solution):
        # 重置PID参数
        num_envs = self.num_envs
        num_solutions = len(input_solution)

        # 循环分配 input_solution 的行到 kPID
        for i in range(num_envs):
            if self.num_input == 4:
                self.kPID[i] = [value * 100 for value in input_solution[i % num_solutions][0:4]]
            else:
                self.kPID[i] = [value * 100 for value in input_solution[i % num_solutions][12:16]]

    def cma_sysID(self, input_solution):
        # 主体函数
        self._reset_envs(input_solution)
        for i in range(int(rotunbot.max_episode_length)):
            self.step()
        # 计算仿真数据和原始数据之间的差异
        output = self.compute_gap(self.sim_data_all, self.raw_data, input_solution)
        return output

    def compute_gap(self, sim_data_all, raw_data, input_solution):
    
        squared_sums = [0.0] * len(input_solution)  # 初始化 squared_sums，与 input_solution 的维度一致

        # 遍历每一维的仿真数据
        for i, sim_data in enumerate(sim_data_all[:len(input_solution)]):  # 只取前 len(input_solution) 维
            squared_sum = 0.0

            # 确保 sim_data 和 raw_data 的长度一致
            min_length = min(len(sim_data), len(raw_data))

            for j in range(min_length):
                sim_row = sim_data[j]
                raw_row = raw_data[j]

                # 计算每个字段的差的平方并累加
                for key in sim_row.keys():
                    if key != "time":  # 跳过时间字段
                        if key == 'yaw':
                            # 对于角度字段，计算差值时需要考虑周期性
                            diff = (sim_row[key] - raw_row[key]) / 5.0
                        elif key == 'first_vel':
                            # 对于速度字段，直接计算差值
                            diff = (sim_row[key] - raw_row[key]) / 1.5
                        elif key == 'second_pos':
                            diff = (sim_row[key] - raw_row[key]) * 10.0 ## 3.0
                        elif key == 'px' or key == 'py':
                            diff = (sim_row[key] - raw_row[key]) / 3.0
                        else:
                            diff = sim_row[key] - raw_row[key]
                        squared_sum += diff ** 2

            # squared_sums[i] = squared_sum * 0.01  # 将平方和存储到对应位置
            squared_sums[i] = squared_sum * 0.1

        return squared_sums  # 返回前 input_solution 维度数量的平方和
    
    def normalize(self, x):
        x = np.array(x)  # 将 x 转换为 NumPy 数组
        search_range = np.array(self.search_range)
        return (x - search_range[0]) / (search_range[1] - search_range[0])

    def denormalize(self, x):
        x = np.array(x)  # 将 x 转换为 NumPy 数组
        search_range = np.array(self.search_range)
        return x * (search_range[1] - search_range[0]) + search_range[0]

class MultiPopulationCMAES:
    """
    一个使用多种群思想来运行CMA-ES的管理器。
    
    参数:
    - objective_function: 需要被最小化的黑盒目标函数。
    - dimension: 目标函数输入的维度。
    - n_populations: 要创建的种群（岛屿）数量。
    - population_size: 每个CMA-ES种群的大小 (lambda)。
    - sigma0: 初始标准差（探索步长）。
    - search_bounds: [min_val, max_val]，用于随机初始化种群的起始点。
    - migration_interval: 整数，每隔多少代进行一次“迁移”。
    """
    def __init__(self, objective_function, dimension, n_populations=5, population_size=10, sigma0=0.5, search_bounds=[-15, 15], migration_interval=25):
        self.objective_function = objective_function
        self.dimension = dimension
        self.n_populations = n_populations
        self.population_size = population_size
        self.sigma0 = sigma0
        self.search_bounds = search_bounds
        self.migration_interval = migration_interval
        
        self.populations = []
        self.global_best_solution = None
        self.global_best_fitness = float('inf')
        
        # 初始化所有种群
        self._initialize_populations()

    def _initialize_populations(self):
        """为每个岛屿创建并初始化一个CMA-ES实例。"""
        print(f"初始化 {self.n_populations} 个种群...")
        for i in range(self.n_populations):
            # 在搜索空间内随机生成一个起始点，以增加多样性
            x0 = np.random.uniform(self.search_bounds[0], self.search_bounds[1], self.dimension)
            
            # CMA-ES的选项
            opts = cma.CMAOptions()
            opts['popsize'] = self.population_size
            opts['bounds'] = self.search_bounds
            opts['verbose'] = -9 # 关闭每个实例的独立输出
            
            # 创建CMA-ES实例并添加到列表中
            es = cma.CMAEvolutionStrategy(x0, self.sigma0, opts)
            self.populations.append(es)
            print(f"  种群 {i+1}/{self.n_populations} 已创建，起始点: {np.round(x0, 2)}")

    def run(self, max_generations=200):
        """
        运行整个优化过程。
        """
        start_time = time.time()
        print(f"\n开始优化过程，最大代数: {max_generations}\n" + "="*40)
        
        for g in range(max_generations):
            all_current_bests = []
            
            # --- 并行进化每个种群 ---
            for i, es in enumerate(self.populations):
                # 1. 'ask' 获取新一代的候选解
                solutions = es.ask()
                
                # 2. 评估每个候选解的适应度（调用黑盒函数）
                fitnesses = [self.objective_function(s) for s in solutions]
                print(solutions)
                print(fitnesses)
                # 3. 'tell' 将解和对应的适应度返回给CMA-ES，以更新其内部状态
                es.tell(solutions, fitnesses)
                
                # 跟踪每个种群当前的最优解
                all_current_bests.append((es.result.fbest, es.result.xbest))

            # --- 更新全局最优解 ---
            current_best_fitness, current_best_solution = min(all_current_bests, key=lambda item: item[0])
            if current_best_fitness < self.global_best_fitness:
                self.global_best_fitness = current_best_fitness
                self.global_best_solution = current_best_solution
            
            # --- 定期进行迁移 ---
            if (g + 1) % self.migration_interval == 0:
                self._perform_migration()

            # --- 打印进度 ---
            if (g + 1) % 10 == 0:
                print(f"代: {g+1:4d} | 全局最优适应度: {self.global_best_fitness:.6f}")

        end_time = time.time()
        print("="*40 + "\n优化完成！")
        print(f"总耗时: {end_time - start_time:.2f} 秒")
        print(f"找到的最优解 (x): {self.global_best_solution}")
        print(f"对应的函数值 (f(x)): {self.global_best_fitness}")
        
        return self.global_best_solution, self.global_best_fitness

    def _perform_migration(self):
        """
        执行迁移操作：
        将全局最优解注入到一个随机选择的、表现较差的种群中，以帮助其跳出局部最优。
        """
        print(f"\n--- 第 {self.populations[0].countiter} 代，执行迁移 ---")
        
        # 找到表现最差的种群（不是必须的，但可以作为一种策略）
        fitnesses = [es.result.fbest for es in self.populations]
        worst_pop_index = np.argmax(fitnesses)
        
        # 将该种群的平均值（下一代采样的中心）重置为已知的全局最优解
        # 这是一种“软”注入，引导该种群向更有希望的区域探索
        print(f"将全局最优解 (适应度: {self.global_best_fitness:.4f}) 注入到种群 {worst_pop_index+1}")
        self.populations[worst_pop_index].mean = self.global_best_solution.copy()
        # 也可以考虑重置其步长，以鼓励在新区域进行更精细的搜索
        # self.populations[worst_pop_index].sigma = self.sigma0 
        print("--- 迁移完成 ---\n")

if __name__ == '__main__':
    rotunbot = SYSID()

    initial_mean = rotunbot.init_solution
    initial_sigma = 0.5
    search_bounds = rotunbot.search_range

    num_populations = 4
    max_iterations_per_pop = 300
    migration_interval = 15
    desired_batch_size = int(rotunbot.num_envs / 4)
    populations = []
    global_best_solution = None
    global_best_fitness = float('inf')
    for i in range(num_populations):
        # 在搜索空间内随机生成一个起始点，以增加多样性
        if i == 0:
            x0 = rotunbot.normalize(initial_mean)
        else :
            # x0 = np.array([np.random.uniform(lower, upper) for lower, upper in zip(search_bounds[0], search_bounds[1])])
            x0 = np.random.uniform(0, 1, rotunbot.num_input)
        # CMA-ES的选项
        opts = cma.CMAOptions()
        opts['popsize'] = desired_batch_size
        opts['bounds'] = [0,1]
        opts['seed'] = i*36 + 7
            
        # 创建CMA-ES实例并添加到列表中
        es = cma.CMAEvolutionStrategy(x0, initial_sigma, opts)
        populations.append(es)
        print(f"  种群 {i+1}/{num_populations} 已创建，起始点: {np.round(x0, 2)}")
    es1 = populations[0]
    es2 = populations[1]
    es3 = populations[2]
    es4 = populations[3]
    start_time = time.time()
    for g in range(max_iterations_per_pop):
        all_current_bests = []
            
        # --- 并行进化每个种群 ---
        
        solutions1 = es1.ask()
        solutions2 = es2.ask()
        solutions3 = es3.ask()
        solutions4 = es4.ask()
        solutions = solutions1 + solutions2 + solutions3 + solutions4
        fitnesses = rotunbot.cma_sysID(rotunbot.denormalize(solutions))
        fitnesses1 = fitnesses[:len(solutions1)]
        fitnesses2 = fitnesses[len(solutions1):len(solutions1) + len(solutions2)]
        fitnesses3 = fitnesses[len(solutions1) + len(solutions2):len(solutions1) + len(solutions2) + len(solutions3)]
        fitnesses4 = fitnesses[len(solutions1) + len(solutions2) + len(solutions3):]

        es1.tell(solutions1, fitnesses1)
        es2.tell(solutions2, fitnesses2)
        es3.tell(solutions3, fitnesses3)
        es4.tell(solutions4, fitnesses4)
        all_current_bests.append((es1.result.fbest, es1.result.xbest))
        all_current_bests.append((es2.result.fbest, es2.result.xbest))
        all_current_bests.append((es3.result.fbest, es3.result.xbest))
        all_current_bests.append((es4.result.fbest, es4.result.xbest))
        # for i, es in enumerate(populations):
        #     # 1. 'ask' 获取新一代的候选解
        #     solutions = es.ask()
                
        #     # 2. 评估每个候选解的适应度（调用黑盒函数）
        #     fitnesses = rotunbot.cma_sysID(solutions)
        #     # 3. 'tell' 将解和对应的适应度返回给CMA-ES，以更新其内部状态
        #     es.tell(solutions, fitnesses)
                
        #     # 跟踪每个种群当前的最优解
        #     all_current_bests.append((es.result.fbest, es.result.xbest))

        # --- 更新全局最优解 ---
        current_best_fitness, current_best_solution = min(all_current_bests, key=lambda item: item[0])
        if current_best_fitness < global_best_fitness:
            global_best_fitness = current_best_fitness
            global_best_solution = current_best_solution
            
            # --- 定期进行迁移 ---
        if (g + 1) % migration_interval == 0:
            print(f"\n--- 第 {populations[0].countiter} 代，执行迁移 ---")
        
            # 找到表现最差的种群（不是必须的，但可以作为一种策略）
            fitnesses = [es.result.fbest for es in populations]
            worst_pop_index = np.argmax(fitnesses)
            
            # 将该种群的平均值（下一代采样的中心）重置为已知的全局最优解
            # 这是一种“软”注入，引导该种群向更有希望的区域探索
            print(f"将全局最优解 (适应度: {global_best_fitness:.4f}) 注入到种群 {worst_pop_index+1}")
            populations[worst_pop_index].mean = global_best_solution.copy()
            # 也可以考虑重置其步长，以鼓励在新区域进行更精细的搜索
            # self.populations[worst_pop_index].sigma = self.sigma0 
            print("--- 迁移完成 ---\n")

        # --- 打印进度 ---
        
        print(f"代: {g+1:4d} | 全局最优适应度: {current_best_fitness:.6f}，解为 {np.round(rotunbot.denormalize(current_best_solution), 4)}")

    end_time = time.time()
    print("="*40 + "\n优化完成！")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"找到的最优解 (x): {global_best_solution}")
    print(f"对应的函数值 (f(x)): {global_best_fitness}")
    
    # es = cma.CMAEvolutionStrategy(initial_mean, initial_sigma, 
    #                               {'popsize': desired_batch_size, # <-- 设定批处理数量
    #                                'seed': 53,
    #                                'tolfun': 1e-6 ,
    #                                'bounds': rotunbot.search_range})
    
    # global_best_solution = None
    # global_best_fitness = float('inf')

    # migration_frequency = 5
    # migration_rate = 0.1 # 每次移民的个体比例
    # current_best_solution = None
    # current_best_fitness = float('inf')

    # for iteration in range(max_iterations_per_pop):
        
    #     # 生成并评估个体
    #     solutions = es.ask()
    #     fitness_values = rotunbot.cma_sysID(solutions)
    #     es.tell(solutions, fitness_values)
            
    #     # 记录当前群体的最佳解 (与之前类似)
    #     current_iter_best_fitness = min(fitness_values)
    #     current_iter_best_solution = solutions[np.argmin(fitness_values)]
    #     if current_iter_best_fitness < current_best_fitness:
    #         current_best_fitness = current_iter_best_fitness
    #         # 找到对应这个最优 fitness 的 solution
    #         current_best_solution = solutions[np.argmin(fitness_values)]  

    #     print(f"迭代 {iteration+1}: 最佳适应度 = {current_iter_best_fitness:.4e}, 解为 {current_iter_best_solution}")
            
    #     if es.stop(): # 检查停止条件
    #             break
            

    # print("\n--- 优化完成 ---")
    # print(f"全局最佳适应度: {current_best_solution:.4e}")
    # print(f"全局最佳解: {current_best_solution}")
    # rotunbot.gym.destroy_viewer(rotunbot.viewer)
    # rotunbot.gym.destroy_sim(rotunbot.sim)
    