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

import matplotlib.pyplot as plt

def plot_joint_data(torque_data, vel_data, pos_data):
    """
    绘制关节数据图像，分成两个子图。

    参数:
    time (list): 时间轴数据
    torque_data (list): 扭矩数据
    vel_data (list): 速度数据
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 6))
    time = np.linspace(0, len(torque_data)*sim_params.dt, len(torque_data))
    # 左边子图：joint1_torque_record
    axs[0].plot(time, torque_data, label='Joint 1 Torque')
    axs[0].set_title('Joint 1 Torque vs Time')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Torque')
    axs[0].legend()
    axs[0].grid(True)

    # 右边子图：joint1_vel_record
    axs[1].plot(time, vel_data, label='Joint 1 Velocity', color='orange')
    axs[1].set_title('Joint 1 Velocity vs Time')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Velocity')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(time, pos_data, label='Joint 1 Position', color='blue')
    axs[2].set_title('Joint 1 Position vs Time')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Position')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

# defination
ball_torque = np.zeros(2)
data_record = False
Torque_Output_index = np.zeros(3)

# def Data_Record():
#     f = open('data/simball_vel9.csv','w',encoding='utf-8')

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Spherical Robot control Example")

device = args.sim_device if args.use_gpu_pipeline else 'cpu'
# 建立仿真器和物理参数
# create a simulator
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.substeps = 2
sim_params.dt = 0.02
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
# set Flex-specific parameters
# sim_params.flex.solver_type = 5
# sim_params.flex.num_outer_iterations = 4
# sim_params.flex.num_inner_iterations = 20
# sim_params.flex.relaxation = 0.8
# sim_params.flex.warm_start = 0.5
# sim_params.use_gpu_pipeline = False
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()
# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')
# 建立地面
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.static_friction = 0.6
plane_params.dynamic_friction = 0.5
plane_params.restitution = 0
gym.add_ground(sim, plane_params)

#建立地形
# plane_width = 50
# plane_length = 50
# terrain_width = 10.
# terrain_length = 10.
# horizontal_scale = 0.1
# vertical_scale = 0.005  # [m]
# num_rows = int(plane_width/horizontal_scale)
# num_cols = int(plane_length/horizontal_scale)
# half_num_rows = int(num_rows/2)
# half_num_cols = int(num_cols/2)
# env_num_rows = int(terrain_width/horizontal_scale)
# env_num_cols = int(terrain_length/horizontal_scale)
# half_env_num_rows = int(env_num_rows/2)
# half_env_num_cols = int(env_num_cols/2)
# heightfield = np.zeros((num_rows, num_cols), dtype=np.int16)
# print(heightfield.shape)
# def new_sub_terrain(): return SubTerrain(width=env_num_rows, length=env_num_rows, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)

# start1_x = half_num_rows - env_num_rows - half_env_num_rows
# end1_x = half_num_rows - half_env_num_rows
# start1_y = half_num_cols - half_env_num_cols
# end1_y = half_num_cols + half_env_num_cols
# start2_x = half_num_rows + half_env_num_rows
# end2_x = half_num_rows + env_num_rows + half_env_num_rows
# start2_y = half_num_cols - half_env_num_cols
# end2_y = half_num_cols + half_env_num_cols
# heightfield[start1_y : end1_y, start1_x : end1_x] = random_uniform_terrain(new_sub_terrain(), min_height=-0.05, max_height=0.04, step=0.03, downsampled_scale=0.2).height_field_raw
# heightfield[start2_y : end2_y, start2_x : end2_x] = pyramid_sloped_terrain(new_sub_terrain(), slope=0.3).height_field_raw
# vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
# tm_params = gymapi.TriangleMeshParams()
# tm_params.nb_vertices = vertices.shape[0]
# tm_params.nb_triangles = triangles.shape[0]
# tm_params.transform.p.x = -25.
# tm_params.transform.p.y = -25.
# gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)

# 键盘输入
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "up")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "left")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "down")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "right")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "stop")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "wheel_left")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_E, "wheel_right")

## 载入球型机器人模型
asset_path = "{LEGGED_GYM_ROOT_DIR}/resources/robots/Rotunbot_wheel/urdf/Rotunbot_wheel.urdf".format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
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
ball_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# 机器人初始姿态
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 0.0, 0.4)
initial_pose.r = gymapi.Quat( 0.0,0.0, 0.0, 1)


# 建立仿真环境
num_envs = 1
spacing = 2.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# 生成多球
# envs = []
# actor_handles = []

# for i in range(num_envs):
#     env = gym.create_env(sim, env_lower, env_upper, 2)
#     envs.append(env)
#     actor_handle = gym.create_actor(env, ball_asset, initial_pose, 'ball', i,1)

#     props = gym.get_asset_dof_properties(ball_asset)
#     props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
#     props["stiffness"].fill(0.0)
#     props["damping"].fill(0.0)
#     gym.set_actor_dof_properties(env, actor_handle, props)
#     actor_handles.append(actor_handle)

# 生成单球
env0 = gym.create_env(sim, env_lower, env_upper, 2)
Spherical_Robot0 = gym.create_actor(env0, ball_asset, initial_pose, 'Rotunbot_wheel', 0,1, 0)

# 关节设置
# 球型机器人关节力矩控制
# props = gym.get_asset_dof_properties(ball_asset)
# props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
# props["stiffness"].fill(0.0)
# props["damping"].fill(0.0)
# gym.set_actor_dof_properties(env0, Spherical_Robot0, props)
# 主轴速度控制 副轴力矩控制
props = gym.get_actor_dof_properties(env0, Spherical_Robot0)
props["driveMode"] = (gymapi.DOF_MODE_EFFORT)
props["stiffness"] = (0.0)
props["damping"] = (0.0)
gym.set_actor_dof_properties(env0, Spherical_Robot0, props)


spherical_robot_num_dofs = gym.get_asset_dof_count(ball_asset)
default_dof_state = np.zeros(spherical_robot_num_dofs, gymapi.DofState.dtype)

gravity_vec = to_torch(get_axis_params(-1., 2), device=device).repeat((num_envs, 1))
print(gravity_vec)
#gym.set_actor_dof_states(env0, Spherical_Robot0, default_dof_state, gymapi.STATE_ALL)

# 记录机器人初始位姿
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

# 读取球壳的id
shell_idx = gym.find_actor_rigid_body_index(env0, Spherical_Robot0, "link1", gymapi.DOMAIN_SIM)
print(shell_idx)
first_axis_idx = gym.find_actor_rigid_body_index(env0, Spherical_Robot0, "base_link", gymapi.DOMAIN_SIM)
print(first_axis_idx)
second_axis_idx = gym.find_actor_rigid_body_index(env0, Spherical_Robot0, "link2", gymapi.DOMAIN_SIM)
print(second_axis_idx)
third_axis_idx = gym.find_actor_rigid_body_index(env0, Spherical_Robot0, "link3", gymapi.DOMAIN_SIM)
print(third_axis_idx)
# tensor读取机器人定位及姿态
gym.prepare_sim(sim)

_Spherical_robot_states = gym.acquire_rigid_body_state_tensor(sim)
Spherical_robot_states = gymtorch.wrap_tensor(_Spherical_robot_states)
_actor_root_state = gym.acquire_actor_root_state_tensor(sim)
actor_root_state = gymtorch.wrap_tensor(_actor_root_state)
_Spherical_robot_dof_states = gym.acquire_dof_state_tensor(sim)
Spherical_robot_dof_states = gymtorch.wrap_tensor(_Spherical_robot_dof_states)
_dof_force_tensor = gym.acquire_dof_force_tensor(sim)
dof_force_tensor = gymtorch.wrap_tensor(_dof_force_tensor).view(num_envs, 3)
print(actor_root_state)
#力矩控制初始化
dof_pos = Spherical_robot_dof_states[:, 0].view(num_envs, 3, 1)
dof_vel = Spherical_robot_dof_states[:, 1].view(num_envs, 3, 1)
actions_tensor = torch.zeros_like(dof_pos).squeeze(-1)
print(Spherical_robot_states.shape)
first_axis_vel = 0
second_axis_pos = 0
flag = 0
count = 0
last_dof_vel = torch.zeros_like(dof_vel)
last_base_lin_vel = torch.zeros_like(actor_root_state[:, 7:10])

# 记录电机的的力矩、角速度
joint1_torque_record = []
joint2_torque_record = []
joint1_vel_record = []
joint2_vel_record = []
joint1_pos_record = []
joint2_pos_record = []

if data_record:
    f=open('data/torque_outputs.csv','w',encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(['torque1','torque2','v'])

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    #更新tensor
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_force_tensor(sim)
    shell_pos = Spherical_robot_states[shell_idx, :3]
    robot_pos = actor_root_state[:, :3]
    
    shell_lin_vel = Spherical_robot_states[shell_idx, 7:10]
    base_quat = actor_root_state[:, 3:7]
    base_quat_array = base_quat.cpu().numpy()
    base_ret = Rotation.from_quat(base_quat_array)
    base_euler = base_ret.as_euler('zyx',degrees=True)
    # 得到结果对应 yaw,roll,pitch
    roll = torch.as_tensor(base_euler[:,1])

    shell_rot = Spherical_robot_states[shell_idx, 3:7]
    shell_rot_array = shell_rot.cpu().numpy()
    shell_ret = Rotation.from_quat(shell_rot_array)
    shell_rot_euler = shell_ret.as_euler('zyx',degrees=True)

    pendulum_rot = Spherical_robot_states[second_axis_idx, 3:7]
    pendulum_rot_array = pendulum_rot.cpu().numpy()
    pendulum_ret = Rotation.from_quat(pendulum_rot_array)
    pendulum_rot_euler = pendulum_ret.as_euler('zyx',degrees=True)

    base_lin_vel = quat_rotate_inverse(base_quat, actor_root_state[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, actor_root_state[:, 10:13])
    projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)

    lin_vel = actor_root_state[:, 7:10]
    base_vel = torch.sqrt(torch.sum(torch.square(lin_vel[:, :2]),dim=1))
    
    # print("vel")
    # print(dof_pos[0][0][0])
    # print(actor_root_state[:, 10:13])
    
    # print("euler")
    # print(projected_gravity)
    # print(robot_pos_orient[0][0])
    # count = count + 1
    # if(count==5):
    #     print(shell_rot_euler)
    #     count = 0
    
    # print("vel")
    # print(base_lin_vel)
    # print(actor_root_state[:, 7:10])
    # print(projected_gravity)
    #主轴-y负方向  ；  副轴-x正方向
    
    # for evt in gym.query_viewer_action_events(viewer):
    #     if evt.action == "reset" and evt.value > 0:
    #         gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
    #         actions_tensor[0,0] = 0
    #         actions_tensor[0,1] = 0

    #     elif evt.action == "up" and evt.value > 0:
    #         actions_tensor[0,0] = 3
        
    #     elif evt.action == "down" and evt.value > 0:
    #         actions_tensor[0,0] = -3

    #     elif evt.action == "left" and evt.value > 0:
    #         actions_tensor[0,1] = 3

    #     elif evt.action == "right" and evt.value > 0:
    #         actions_tensor[0,1] = -3

    #     elif evt.action == "stop" and evt.value > 0:
    #         actions_tensor[0,0] = 0
    #         actions_tensor[0,1] = 0

    #接收力矩值
    # actions_tensor[0,0] = ball_torque[0]
    # actions_tensor[0,1] = ball_torque[1]

    #主轴速度控制 力矩和前进方向相反

    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
            first_axis_vel = 0
            second_axis_pos = 0

        elif evt.action == "up" and evt.value > 0:
            first_axis_vel = 6.5 # 5---100
            
        
        elif evt.action == "down" and evt.value > 0:
            first_axis_vel = -1

        elif evt.action == "left" and evt.value > 0:
            second_axis_pos = -5

        elif evt.action == "right" and evt.value > 0:
            second_axis_pos = 0.8 #1.4---70
            count = 0
            # flag = 1
            
        elif evt.action == "wheel_left" and evt.value > 0:
            actions_tensor[0,2] = 5 
            
        elif evt.action == "wheel_right" and evt.value > 0:
             actions_tensor[0,2] = -5

        elif evt.action == "stop" and evt.value > 0:
            first_axis_vel = 0
            second_axis_pos = 0
            actions_tensor[0,2] = 0
    
    if flag == 1:
        actions_tensor[0,1] = second_axis_pos
        joint1_vel_record.append(dof_vel[0, 0].item())
        joint2_vel_record.append(dof_vel[0, 1].item())
        joint1_torque_record.append(actions_tensor[0,0].item())
        joint2_torque_record.append(actions_tensor[0, 1].item())
        joint1_pos_record.append(dof_pos[0, 0].item())
        joint2_pos_record.append(dof_pos[0, 1].item())
        count += 1
    if count == 600:
        plot_joint_data(joint2_torque_record, joint2_vel_record, joint2_pos_record)
        flag = 0 
        count = 0
        joint1_vel_record.clear()
        joint2_vel_record.clear()
        joint1_torque_record.clear()
        joint2_torque_record.clear()
        joint1_pos_record.clear()
        joint2_pos_record.clear()
        
    #主轴速度控制
    actions_tensor[0,0] =  25 * (first_axis_vel - dof_vel[0][0][0]) - 1 * (dof_vel[0][0][0] - last_dof_vel[0][0][0]) / sim_params.dt
    # actions_tensor[0,0] =  first_axis_vel
    #副轴位置控制
    if flag == 0:
        actions_tensor[0,1] = 15 * ( second_axis_pos  - dof_pos[0][1][0]) - 5 * dof_vel[0][1][0]
    # actions_tensor[0,1] = 15 * ( second_axis_pos  - dof_pos[0][1][0]) - 5 * dof_vel[0][1][0]
    
    # actions_tensor[0,1] = second_axis_pos
    torques = torch.clip(actions_tensor,-400.,400.) 
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torques))
    # print('---------------')
    # print(torques)
    # print(base_lin_vel)
    if data_record:
         Torque_Output_index[0] = torques[:,0]
         Torque_Output_index[1] = torques[:,1]
         Torque_Output_index[2] = dof_vel[0][0][0]
         writer.writerow(Torque_Output_index)
    last_dof_vel[:] = dof_vel[:]
    last_base_lin_vel[:] = base_lin_vel[:]
    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
