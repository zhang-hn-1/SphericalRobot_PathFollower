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

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()
    
    def plot_all_state(self):
        self.plot_process = Process(target=self._plot_state)
        self.plot_process.start()
    
    def plot_trajectories(self):
        self.plot_process = Process(target=self._plot_tra)
        self.plot_process.start()

    def _plot(self):
        nb_rows = 3
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot joint targets and measured positions
        a = axs[1, 0]
        if log["dof_pos"]: a.plot(time, log["dof_pos"], label='measured')
        if log["dof_pos_target"]: a.plot(time, log["dof_pos_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        a.legend()
        # plot joint velocity
        a = axs[1, 1]
        if log["dof_vel"]: a.plot(time, log["dof_vel"], label='measured')
        if log["dof_vel_target"]: a.plot(time, log["dof_vel_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()
        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend()
        # plot base vel y
        a = axs[0, 1]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.legend()
        # plot base vel yaw
        a = axs[0, 2]
        if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        a.legend()
        # plot base vel z
        a = axs[1, 2]
        if log["base_vel_z"]: a.plot(time, log["base_vel_z"], label='measured')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
        a.legend()
        # plot contact forces
        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f'force {i}')
        a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        a.legend()
        # plot torque/vel curves
        a = axs[2, 1]
        if log["dof_vel"]!=[] and log["dof_torque"]!=[]: a.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured')
        a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
        a.legend()
        # plot torques
        a = axs[2, 2]
        if log["dof_torque"]!=[]: a.plot(time, log["dof_torque"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()
        plt.show()
    
    def _plot_state(self):
        """
        绘制关节数据图像，分成两个子图。

        参数:
        time (list): 时间轴数据
        torque_data (list): 扭矩数据
        vel_data (list): 速度数据
        """
        fig, axs = plt.subplots(4, 2, figsize=(12, 6))
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # 左边子图：joint1_torque_record
        a = axs[0,0]
        a.plot(time, log["base_vel_x"], label='measured')
        a.plot(time, log["command_x"], label='commanded')
        # a.set_xlabel('Time')
        # a.set_ylabel('Torque')
        a.legend()
        a.grid(True)

        # 右边子图：joint1_vel_record
        a = axs[0,1]
        a.plot(time, log["dof_vel"], label='measured')
        a.plot(time, log["dof_vel_target"], label='target')
        a.set_title('Joint Velocity vs Time')
        a.set_xlabel('Time')
        a.set_ylabel('Velocity')
        a.legend()
        a.grid(True)

        a = axs[1,0]
        a.plot(time, log["dof_pos"], label='measured')
        a.plot(time, log["dof_pos_target"], label='target')
        a.set_title('Joint 2 Position vs Time')
        a.set_xlabel('Time')
        a.set_ylabel('Position')
        a.legend()
        a.grid(True)

        a = axs[1,1]
        a.plot(time, log["base_vel_yaw"], label='measured')
        a.plot(time, log["command_yaw"], label='commanded')
        # a.set_title('Quat vs Time')
        # a.set_xlabel('Time')
        # a.set_ylabel('Quat')
        a.legend()
        a.grid(True)

        a = axs[2,0]
        a.plot(time, log['ang_vel_x'], label='Base Ang Vel x', color='green')
        a.plot(time, log['ang_vel_y'], label='Base Ang Vel y', color='red')
        a.plot(time, log['ang_vel_z'], label='Base Ang Vel z', color='blue')
        a.set_title('Ang Vel vs Time')
        a.set_xlabel('Time')
        a.set_ylabel('Ang Vel')
        a.legend()
        a.grid(True)

        a = axs[2,1]
        a.plot(time, log['base_vel_x'], label='Lin Vel x', color='green')
        a.plot(time, log['base_vel_y'], label='Lin Vel y', color='red')
        a.plot(time, log['base_vel_z'], label='Lin Vel z', color='blue')
        a.set_title('Lin Vel vs Time')
        a.set_xlabel('Time')
        a.set_ylabel('Lin Vel')
        a.legend()
        a.grid(True)

        a = axs[3,0]
        # a.plot(time, base_pos_data, label='Position')
        # a.set_title('Position vs Time')
        # a.set_xlabel('Time')
        # a.set_ylabel('Position')
        a.legend()
        a.grid(True)

        a = axs[3,1]
        a.plot(time, log['roll'], label='Roll', color='green')
        a.plot(time, log['pitch'], label='Pitch', color='red')
        a.plot(time, log['yaw'], label='Yaw', color='blue')
        a.set_title('Euler vs Time')
        a.set_xlabel('Time')
        a.set_ylabel('Euler')
        a.legend()
        a.grid(True)

        plt.tight_layout()
        plt.show()
    
    def _plot_tra(self):
        nb_rows = 2
        nb_cols = 2

        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot joint targets and measured positions
        a = axs[1, 0]
        if log["goal_dist"]!=[]: a.plot(time, log["goal_dist"], label='measured')
        a.set(xlabel='time [s]', ylabel='distance to target [m]', title='Distance')
        a.legend()
        # plot joint velocity
        a = axs[1, 1]
        if log["dof_pos"]!=[]: a.plot(time, log["dof_pos"], label='measured')
        a.set(xlabel='time [s]', ylabel='dof 2 pos', title='dof 2 position')
        a.legend()
        # plot trajectory
        a = axs[0, 0]
        if log["base_pos_x"]: a.plot(log["base_pos_x"], log["base_pos_y"], label='measured')
        if log["ref_pos_x"]: a.plot(log["ref_pos_x"], log["ref_pos_y"], label='commanded')
        a.set(xlabel='x [m]', ylabel='y [m]', title='Trajectory')
        a.legend()
        # plot base vel y
        a = axs[0, 1]
        if log["dof_vel"]: a.plot(time, log["dof_vel"], label='measured')
        a.set(xlabel='time [s]', ylabel='dof 1 vel [m/s]', title='dof 1 velocity')
        a.legend()
        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()