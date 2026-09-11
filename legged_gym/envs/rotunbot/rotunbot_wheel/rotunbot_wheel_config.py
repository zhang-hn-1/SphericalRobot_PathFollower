from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class RotunbotWheelCfg( LeggedRobotCfg ):
    """
    Configuration class for the Rotunbot spherical robot.
    """
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_actions = 3
        num_observations = 29#17#+6
        # frame_stack = 66      #all histroy obs num
        # short_frame_stack = 5   #short history step
        # c_frame_stack = 3  #all histroy privileged obs num
        # num_single_obs = 47
        # num_observations = int(frame_stack * num_single_obs)
        # single_num_privileged_obs = 73
        # single_linvel_index = 53
        # num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        # num_actions = 12
        # episode_length_s = 24 #episode length in seconds
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
            'joint3': 0.0,
        }

    class control( LeggedRobotCfg.control ):
        control_type = 'T' # P: position, V: velocity, T: torques
        # action scale: target torques = actionScale * action
        action_scale = 10
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        torque_limits_1 = 25
        torque_limits_2 = 15
        torque_limits_3 = 5
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
            lin_vel_y = [-2.0, 2.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
    
    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/Rotunbot_wheel/urdf/Rotunbot_wheel.urdf"
        name = "Rotunbot_wheel"

        terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"]
        self_collisions = 1 
        

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        push_robots = True
  
    class rewards( LeggedRobotCfg.rewards ):
        base_height_target = 0.5
        max_contact_force = 500.
        only_positive_rewards = True
        class scales( LeggedRobotCfg.rewards.scales ):
            pass
    
    class normalization( LeggedRobotCfg.normalization ):
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 0.2 # scales other values
        class noise_scales:
            quat = 0.1
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.5
            tracking_ang_vel = 2.0
            lin_vel_z = -2.0 # 减小高度奖励
            ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            torques = -0.00001 # 减小力矩奖励
            dof_acc = -2.5e-7 # 减小加速度奖励
            action_rate = -0.01 # 减小动作速率奖励

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        


class RotunbotWheelCfgPPO(LeggedRobotCfgPPO):
    seed = 3
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.2
        actor_hidden_dims = [1024, 512, 256, 128]
        critic_hidden_dims = [1024, 512, 256, 128]
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
        experiment_name = 'rotunbot_wheel'
        num_steps_per_env = 200
        max_iterations = 500  # number of policy updates
        
