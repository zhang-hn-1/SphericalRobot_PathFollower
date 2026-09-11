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

import cma
    

class SYSID():

    def __init__(self):
        self._init_parameter()
        self._init_env(headless=False)
        self._init_buffers()
        self._reset_envs(self.init_solution)
    
    def _init_parameter(self):

        self.num_envs = 1
        self.num_input = 16 # 16 # 25

        # 初始化运行时间
        episode_length_s = 12.0 #12.0

        self.dt = 0.02
        self.max_episode_length = np.ceil(episode_length_s / self.dt)
        self.episode_length_buf = 0
        # 初始化参数 [m1, m2, m3, Ixx0, Iyy0, Izz0, Ixx1, Iyy1, Izz1, Ixx2, Iyy2, Izz2, Kp1, Kd1, Kp2, Kd2]
        #                                                                  9    10   11   12
        self.init_solution = [30.225, 27.834, 73.4, 1.1392, 1.097461, 1.598825, 2.3603, 1.8803, 2.3606, 0.774154, 1.782475, 1.490523, 0.35, 0.0, 2.0, 1.0]
        # self.init_solution = [0.7582,0.5609,3.9965,3.0492]
        self.kPID = [35.0, 0.0, 200.0, 100.0]
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
        self.search_range = [
            [25.0, 20.0, 65.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  0.0,  0.1,  0.1],  # 初始值
            [35.0, 35.0, 85.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 3.0, 2.0]  # 截止值
        ]

        # 初始化数据记录
        
        self.file_path = "data/state_data_raw_v.txt"
        self.sim_data = []
        self.raw_data = self.read_and_process_state_data(self.file_path,episode_length_s,self.dt)

    def log_state(self, key, value):
        self.sim_data[key].append(value)

    def log_states(self, dict):
        self.sim_data.append(dict)

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
            for row in data:
                row["time"] = round(row["time"] - initial_time, 2)

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
        
    def read_and_process_state_data(self, file_path, total_time, time_interval):
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
            row["yaw"] = (row["yaw"] - initial_yaw + math.pi) % (2 * math.pi) - math.pi
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
            body_props = self._process_rigid_body_props(body_props, self.init_solution)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
        
        self.gym.prepare_sim(self.sim)
        
    def _process_rigid_body_props(self, props, input_solution):
        if self.num_input == 4:
            return props
        
        for i in range(len(props)):
            props[i].mass = input_solution[i]  # m
            props[i].inertia.x.x = input_solution[i * 3 + 3]  # Ixx
            props[i].inertia.y.y = input_solution[i * 3 + 4]  # Iyy
            props[i].inertia.z.z = input_solution[i * 3 + 5]  # Izz
                
        return props

    def step(self):
        # 执行一步仿真
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
        robot_index = 0
        self.log_states(
            {
                'time': self.episode_length_buf * self.dt,
                # 'px': self.root_states[robot_index, 0].item(),
                # 'py': self.root_states[robot_index, 1].item(),
                'roll': self.base_euler_tensor[robot_index, 0].item(),
                'pitch': self.base_euler_tensor[robot_index, 1].item(),
                'yaw': self.base_euler_tensor[robot_index, 2].item(),
                'vx': self.base_lin_vel[robot_index, 0].item(),
                'vy': self.base_lin_vel[robot_index, 1].item(),
                'vz': self.base_lin_vel[robot_index, 2].item(),
                'wx': self.base_ang_vel[robot_index, 0].item(),
                'wy': self.base_ang_vel[robot_index, 1].item(),
                'wz': self.base_ang_vel[robot_index, 2].item(),
                'second_pos': self.dof_pos[robot_index, 1].item(),
                'first_vel': self.dof_vel[robot_index, 0].item()
            }
        )
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

    def get_action_sequence(self, episode_length_buf):
        """
        根据时间计数生成动作序列。
        参数:
            episode_length_buf (int): 时间计数，每 0.02 秒记一次。
        返回:
            float: 当前时间点的动作值。
        """
        # 将时间计数转换为秒
        time = episode_length_buf * self.dt  # self.dt = 0.02

        if 2.0 <= time < 3.0:
            # 第 2 秒到第 3 秒，生成阶跃信号
            action1 = 2.0
        else:
            # 其他时间，动作为 0
            action1 = 0.0

        # 动作序列逻辑
        if time < 1.0:
            action2 = 0.0
        elif 1.0 <= time < 5.0:
            action2 = 2.0
        elif 5.0 <= time < 7.0:
            action2 = 2.0 - (4.0 / 2.0) * (time - 5.0)  # 线性下降
        elif 7.0 <= time < 8.0:
            action2 = -2.0
        elif 8.0 <= time < 10.0:
            action2 = -2.0 + (2.0 / 2.0) * (time - 8.0)  # 线性上升
        elif 10.0 <= time < 20.0:
            action2 = 3.0 * np.cos(2 * np.pi * (time - 12.5) / 10.0)  # 余弦信号
        else:
            action2 = 0.0

        # 生成第二个动作
        

        return [action1,action2]
    
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

        torques[:,0] =  kPID[0] * (actions[:,0] - self.dof_vel[:,0]) - kPID[1]*(self.dof_vel[:,0] - self.last_dof_vel[:,0])/self.dt
        torques[:,1] =  kPID[2] * (actions[:,1]  - self.dof_pos[:,1]) - kPID[3] * self.dof_vel[:,1]

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
        self.kPID = np.zeros((2, 2), dtype=np.float32)

    def _reset_envs(self, input_solution):
        # 重置环境
        self.episode_length_buf = 0
        self.sim_data.clear()
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
            body_props = self._process_rigid_body_props(body_props, input_solution)
            self.gym.set_actor_rigid_body_properties(env, actor_handle, body_props, recomputeInertia=True)
            body_props = self.gym.get_actor_rigid_body_properties(env, actor_handle)
    
    def reset_kPID(self, input_solution):
        # 重置PID参数
        if self.num_input == 4:
            self.kPID = [value * 100 for value in input_solution[0:4]]
        else:
            self.kPID = [value * 100 for value in input_solution[12:16]]

    def cma_sysID(self, input_solution):
        # 主体函数
        self._reset_envs(input_solution)
        for i in range(int(rotunbot.max_episode_length)):
            self.step()
        # 计算仿真数据和原始数据之间的差异
        output = self.compute_gap(self.sim_data, self.raw_data)
        return output

    def compute_gap(self, sim_data, raw_data):
        # 计算仿真数据和原始数据之间的差异
        squared_sum = 0.0

        # 确保 sim_data 和 raw_data 的长度一致
        min_length = min(len(sim_data), len(raw_data))

        for i in range(min_length):
            sim_row = sim_data[i]
            raw_row = raw_data[i]

            # 计算每个字段的差的平方并累加
            for key in sim_row.keys():
                if key != "time":  # 跳过时间字段
                    if key == 'yaw':
                        # 对于角度字段，计算差值时需要考虑周期性
                        diff = (sim_row[key] - raw_row[key]) / 3.0
                    elif key == 'first_vel':
                        # 对于速度字段，直接计算差值
                        diff = (sim_row[key] - raw_row[key]) / 2.0
                    else:
                        diff = sim_row[key] - raw_row[key]
                    squared_sum += diff ** 2

        return squared_sum
    
    def normalize(self, x, bounds):
        return (x - bounds[0]) / (bounds[:, 1] - bounds[:, 0])

    def denormalize(self, x, bounds):
        return x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]


    def plot_result(self, input_solution):
        self._reset_envs(input_solution)
        for i in range(int(rotunbot.max_episode_length)):
            self.step()
        # 计算仿真数据和原始数据之间的差异
        differences = self.calculate_differences(self.sim_data, self.raw_data)
        for key, value in differences.items():
            print(f"{key}: 最大差异 = {value['max_diff']:.4f}, RMSE = {value['rmse']:.4f}")
        self.plot_data_comparison(self.sim_data, self.raw_data)

    def plot_data_comparison(self, sim_data, raw_data, keys_to_plot=None):
        """
        使用子图绘制 sim_data 和 raw_data 的对比图，展示差异。
        参数:
            sim_data (list): 仿真数据，每个元素是一个字典。
            raw_data (list): 原始数据，每个元素是一个字典。
            keys_to_plot (list): 要绘制的字段列表。如果为 None，则绘制所有字段。
        """
        # 确保 sim_data 和 raw_data 的长度一致
        min_length = min(len(sim_data), len(raw_data))
        sim_data = sim_data[:min_length]
        raw_data = raw_data[:min_length]

        # 获取两者都包含的字段
        common_keys = set(sim_data[0].keys()).intersection(set(raw_data[0].keys()))
        if keys_to_plot is None:
            keys_to_plot = [key for key in common_keys if key != "time"]

        # 提取时间序列
        time = [row["time"] for row in sim_data]

        # 创建 6×2 的子图
        fig, axes = plt.subplots(5, 3, figsize=(16, 12), sharex=True)
        axes = axes.flatten()  # 将 6×2 的子图展平为一维数组

        # 绘制每个字段的对比图
        for i, key in enumerate(keys_to_plot):
            if i >= len(axes):  # 如果字段数量超过子图数量，跳过
                break
            sim_values = [row[key] for row in sim_data]
            raw_values = [row[key] for row in raw_data]

            axes[i].plot(time, sim_values, label=f"Simulated {key}", linestyle='--')
            axes[i].plot(time, raw_values, label=f"Raw {key}", linestyle='-')
            axes[i].set_title(f"Comparison of {key}")
            axes[i].set_ylabel(key)
            axes[i].grid(True)
            axes[i].legend()

        # 隐藏未使用的子图
        for j in range(len(keys_to_plot), len(axes)):
            axes[j].axis('off')

        # 设置共享的 x 轴标签
        axes[-1].set_xlabel("Time (s)")

        # 调整布局
        plt.tight_layout()

        # 显示图像
        plt.show()

    def calculate_differences(self, sim_data, raw_data):
        """
        计算每种参数的 raw_data 和 sim_data 差异的最大值和 RMSE 值。
        参数:
            sim_data (list): 仿真数据，每个元素是一个字典。
            raw_data (list): 真实数据，每个元素是一个字典。
        返回:
            dict: 每种参数的最大差异值和 RMSE 值。
        """
        # 确保 sim_data 和 raw_data 的长度一致
        min_length = min(len(sim_data), len(raw_data))
        sim_data = sim_data[:min_length]
        raw_data = raw_data[:min_length]

        # 获取两者都包含的字段
        common_keys = set(sim_data[0].keys()).intersection(set(raw_data[0].keys()))
        common_keys.discard("time")  # 排除时间字段

        # 初始化结果字典
        results = {}

        # 计算每种参数的差异
        for key in common_keys:
            sim_values = np.array([row[key] for row in sim_data])
            raw_values = np.array([row[key] for row in raw_data])

            # 差异
            differences = sim_values - raw_values

            # 最大差异值
            max_diff = np.max(np.abs(differences))

            # RMSE 值
            rmse = np.sqrt(np.mean(differences ** 2))

            # 保存结果
            results[key] = {"max_diff": max_diff, "rmse": rmse}

        return results

if __name__ == '__main__':
    rotunbot = SYSID()
    input_solution1 = [2.50003e+01,2.00440e+01,8.49508e+01,1.40960e+00,1.08400e-01,1.00400e-01,
                      1.00000e-01,1.70700e-01,1.06700e-01,1.61820e+00,8.26500e-01,1.00800e-01,
                      1.95900e-01,8.60000e-03,7.99600e-01,5.52200e-01]
    input_solution = [2.50012e+01,2.00194e+01,8.48159e+01,2.32390e+00,6.03700e-01,1.61480e+00,
                      4.41800e-01,3.22400e-01,2.97400e+00,1.22900e-01,2.85700e-01,2.99970e+00,
                      2.11700e-01,9.70000e-03,2.97460e+00,1.49970e+00]
    # input_solution = [33.1026,20.035 ,65.155, 0.2775, 2.9869, 0.1004, 0.2522, 0.2629, 0.4522,
    #     2.9845, 2.9383, 0.2128, 1.3646, 1.6273, 2.9998, 0.9882]
    input_solution = [34.9984,34.9906,65.0025, 1.2249, 2.9977, 0.1013, 0.1013, 0.7721, 2.8647,
                      2.999, 2.5727, 0.1005, 0.9135, 0.1485, 2.9996, 1.9998]
    
    # input_solution = [0.7582,0.5609,3.9965,3.0492]
    
    rotunbot.plot_result(input_solution)

    # x0 = rotunbot.init_solution
    # sigma0 = 0.5
    # print(rotunbot.search_range)
    # options = {
    #     'maxfevals': 1000,    # 最大函数评估次数，限制仿真总次数
    #     'bounds': rotunbot.search_range, # 设置每个参数的上下界 [电流下限, 电压下限], [电流上限, 电压上限]
    #     'ftarget': 10.0,      # 如果目标值降到 0.1 以下就停止 (可能不会达到)
    #     'seed': 40           # 设置随机数种子，确保结果可重现
    # }
    # xbest, es = cma.fmin2(rotunbot.cma_sysID, x0, sigma0, options=options)
    # print(xbest)
    
    # rotunbot.gym.destroy_viewer(rotunbot.viewer)
    # rotunbot.gym.destroy_sim(rotunbot.sim)
    