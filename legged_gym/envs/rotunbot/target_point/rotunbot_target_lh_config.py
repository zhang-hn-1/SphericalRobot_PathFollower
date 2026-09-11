from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class RotunbotTargetLHCfg( LeggedRobotCfg ):
    """
    Configuration class for the Rotunbot spherical robot.
    """
    class env( LeggedRobotCfg.env ):
        num_envs = 2048
        num_actions = 2
        # num_observations = 37#17#+6
        frame_stack = 10      #all histroy obs num
        short_frame_stack = 2   #short history step
        c_frame_stack = 3  #all histroy privileged obs num
        num_single_obs = 18
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 21
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        episode_length_s = 100 #episode length in seconds
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

    class sim( LeggedRobotCfg.sim ):
        dt =  0.02 #200Hz 50Hz

    class control( LeggedRobotCfg.control ):
        control_type = 'R' # P: position, V: velocity, T: torques  R:rotunbot
        # action scale: target torques = actionScale * action
        action_scale = 40
        decimation = 2
        # decimation: Number of control action updates @ sim DT per policy DT
        first_actionScale = 1
        second_actionScale = 0.5
        first_vel_limits = 3
        second_pos_limits = 0.45
        set_a_rate_limit = True
        rate_limit_1 = 0.02
        rate_limit_2 = 0.008
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
        num_commands = 3 # default: lin_vel_y, ang_vel_yaw
        resampling_time = 30. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        command_yaw = False # if true: compute yaw command from heading error
        stop_distance = 0.2 # (0.1)distance to target before stopping
        class ranges( LeggedRobotCfg.commands.ranges ):
            pos_x = [-5, 5] # min max [m]
            pos_y = [-5, 5] # min max [m]
            yaw = [-3.14, 3.14] # min max [rad]
            lin_vel_x = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
    
    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/Rotunbot/urdf/Rotunbot.urdf"
        name = "Rotunbot"

        terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"]
        self_collisions = 1 
        

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_base_mass = False
        added_mass_range = [-5., 5.]
        push_robots = False
  
    class rewards( LeggedRobotCfg.rewards ):
        base_height_target = 0.5
        max_contact_force = 500.
        only_positive_rewards = True
        class scales( LeggedRobotCfg.rewards.scales ):
            pass
    
    class normalization( LeggedRobotCfg.normalization ):
        class obs_scales:
            command = 1.0
            lin_vel = 1.0
            ang_vel = 0.5
            quat = 1.0
            dof_pos = 2.0
            dof_vel = 1.0
            pos = 0.2
        clip_observations = 100.
        clip_actions = 100.
    

    class noise:
        add_noise = True
        noise_level = 0.2 # scales other values
        class noise_scales:
            quat = 0.1
            dof_pos = 0.05
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            height_measurements = 0.1
            pos = 0.08

    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 0.2 
            # close_to_orientation = 2.0 
            to_target = 1
            # to_orientation = 2
            stop = 50.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            # balance = -0.004
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            # dof_acc = -2.5e-5 # 减小加速度奖励
            # action_rate = -0.002 # 减小动作速率奖励
            # overturn = -0.4

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        close_para = 1
        tracking_sigma_main = 2 # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 2
        tracking_sigma = 0.5 # tracking reward = exp(-error^2/sigma)
        


class RotunbotTargetLHCfgPPO(LeggedRobotCfgPPO):
    seed = 3
    runner_class_name = 'LHOnPolicyRunner'
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.2
        #actor_hidden_dims = [1024, 512, 256, 128]
        #critic_hidden_dims = [1024, 512, 256, 128]
        actor_hidden_dims = [256, 64, 32]
        critic_hidden_dims = [256, 64, 32]
        # actor_hidden_dims = [512, 256]
        # critic_hidden_dims = [512, 256]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        kernel_size=[3, 2]
        filter_size=[16, 8]
        stride_size=[1, 1]
        lh_output_dim=16   #long history output dim
        in_channels = RotunbotTargetLHCfg.env.frame_stack
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCriticLH'
        algorithm_class_name = 'PPOLH'
        run_name = ''
        experiment_name = 'rotunbot_target_lh'
        num_steps_per_env = 100
        max_iterations = 10000  # number of policy updates
        # resume = True
        # load_run = '/data/lzq/workspace/SphericalRobot_LeggedGym-master/logs/rotunbot_target_lh/Jul24_20-11-33_' # -1 = last run
        # checkpoint = '2000' # -1 = last saved model
        # resume = True
        # load_run = '/home/an/legged_gym/logs/rotunbot_target/Jul25_10-51-54_' # -1 = last run
        # checkpoint = '2250' # -1 = last saved model
        
