from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class RotunbotVelCfg( LeggedRobotCfg ):
    """
    Configuration class for the Rotunbot spherical robot.
    """
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_actions = 2
        # num_observations = 26#17#+6
        num_observations = 15
        num_privileged_obs = 20
        # num_actions = 12
        episode_length_s = 20 #episode length in seconds
        # use_ref_actions = False
        

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class init_state( LeggedRobotCfg.init_state ):
        #initial position
        pos = [0.0, 0.0, 0.4] # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'joint1': 0.0,
            'joint2': 0.0,
        }

    class control( LeggedRobotCfg.control ):
        control_type = 'R' # P: position, V: velocity, T: torques
        # action scale: target torques = actionScale * action
        action_scale = 40
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 2 ## 2
        first_actionScale = 1.0
        second_actionScale = 0.5  ## 0.5
        torque_limits_1 = 100
        torque_limits_2 = 100
        # ********速度控制******** 
        # control_type = 'V' # P: position, V: velocity, T: torques
        # # action scale: target torques = actionScale * action
        # action_scale = 10
        # # decimation: Number of control action updates @ sim DT per policy DT
        # # 需要在rotunbot.py中使用这些参数
        # decimation = 4

    class commands( LeggedRobotCfg.commands ):
        num_commands = 2 # default: lin_vel_y, ang_vel_yaw
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_y = [-0.8, 0.8]   # min max [m/s] ## 1.5
            ang_vel_yaw = [-0.6, 0.6]    # min max [rad/s]
    
    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/Rotunbot/urdf/Rotunbot_test2.urdf"
        name = "Rotunbot"

        terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"]
        self_collisions = 1 
        
    class sim( LeggedRobotCfg.sim ):
        dt =  0.04 #200Hz 50Hz

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_base_mass = False
        added_base_mass_range = [-8., 8.]

        randomize_link_mass = False
        added_link_mass_range = [0.9, 1.1]

        randomize_com = False
        com_displacement_range = [[-0.05, 0.05],
                                  [-0.05, 0.05],
                                  [-0.05, 0.05]]

        randomize_link_com = False
        link_com_displacement_range = [[-0.005, 0.005],
                                  [-0.005, 0.005],
                                  [-0.005, 0.005]]
        
        randomize_base_inertia = False
        base_inertial_range = [[0.98,1.02],
                               [0.98,1.02],
                               [0.98,1.02]]
        
        randomize_link_inertia = False
        link_inertial_range = [[0.98,1.02],
                               [0.98,1.02],
                               [0.98,1.02]]

        push_robots = False

        randomize_friction = False
        friction_range = [0.5, 1.25]
        restitution_range = [0.0, 0.4]


        randomize_torque = False
        torque_multiplier_range = [0.8, 1.2]

        randomize_joint_friction = False
        randomize_joint_friction_each_joint = False
        joint_friction_range = [0.01, 1.15]

        randomize_joint_armature = False
        randomize_joint_armature_each_joint = False
        joint_armature_range = [0.0001, 0.05]

        add_dof_lag = False                # 这个是接收信号（dof_pos和dof_vel)的延迟,dof_pos 和dof_vel延迟一样
        randomize_dof_lag_timesteps = True
        randomize_dof_lag_timesteps_perstep = False  # 不常用always False
        dof_lag_timesteps_range = [2, 15]

        add_imu_lag = False                    # 这个是 imu 的延迟
        randomize_imu_lag_timesteps = True
        randomize_imu_lag_timesteps_perstep = False         # 不常用always False
        imu_lag_timesteps_range = [2, 15]

  
    class normalization( LeggedRobotCfg.normalization ):
        class obs_scales:
            command = 5.0
            lin_vel = 1.0
            ang_vel = 1.0
            quat = 1.0
            dof_pos = 2.0
            dof_vel = 0.5
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            quat = 0.3
            dof_pos = 0.12
            dof_vel = 0.9
            lin_vel = 0.3
            ang_vel = 0.3
            gravity = 0.05
            height_measurements = 0.1

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            # lin_vel_z = -0.002 # 减小高度奖励
            # ang_vel_xy = -2.0 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            # dof_acc = -2.5e-4 # 减小加速度奖励
            # action_rate = -15 # 减小动作速率奖励

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 2.5 # tracking reward = exp(-error^2/sigma)
        


class RotunbotVelCfgPPO(LeggedRobotCfgPPO):
    seed = 4
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.2
        actor_hidden_dims = [64 , 32]
        critic_hidden_dims = [128, 64]
        # actor_hidden_dims = [512, 256]
        # critic_hidden_dims = [512, 256]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rotunbot_vel'
        num_steps_per_env = 50
        max_iterations = 1000  # number of policy updates
        
