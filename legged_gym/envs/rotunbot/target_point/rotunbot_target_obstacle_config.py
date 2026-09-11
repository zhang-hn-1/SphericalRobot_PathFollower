from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class RotunbotTargetObstacleCfg( LeggedRobotCfg ):
    """
    Configuration class for the Rotunbot spherical robot.
    """
    class env( LeggedRobotCfg.env ):
        num_envs = 2048
        num_actions = 2
        # num_observations = 37#17#+6
        num_observations = 19
        # frame_stack = 66      #all histroy obs num
        # short_frame_stack = 5   #short history step
        # c_frame_stack = 3  #all histroy privileged obs num
        # num_single_obs = 4720
        # num_observations = int(frame_stack * num_single_obs)
        # single_num_privileged_obs = 73
        # single_linvel_index = 53
        # num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        # num_actions = 12
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

    class control( LeggedRobotCfg.control ):
        control_type = 'R' # P: position, V: velocity, T: torques  R:rotunbot
        # action scale: target torques = actionScale * action
        action_scale = 40
        # decimation: Number of control action updates @ sim DT per policy DT
        first_actionScale = 1
        second_actionScale = 0.5
        torque_limits_1 = 600
        torque_limits_2 = 600
        # ********速度控制********
        # control_type = 'V' # P: position, V: velocity, T: torques
        # # action scale: target torques = actionScale * action
        # action_scale = 10
        # # decimation: Number of control action updates @ sim DT per policy DT
        # # 需要在rotunbot.py中使用这些参数
        # decimation = 4

    class commands( LeggedRobotCfg.commands ):
        num_commands = 3 # default: lin_vel_y, ang_vel_yaw
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        command_yaw = False # if true: compute yaw command from heading error
        stop_distance = 0.06 # (0.1)distance to target before stopping
        class ranges( LeggedRobotCfg.commands.ranges ):
            pos_x = [-8, 8] # min max [m]
            pos_y = [-8, 8] # min max [m]
            yaw = [-3.14, 3.14] # min max [rad]
            lin_vel_x = [-0.8, 0.8]   # min max [m/s]
            ang_vel_yaw = [-0.4, 0.4]    # min max [rad/s]
    
    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/Rotunbot/urdf/Rotunbot.urdf"
        name = "Rotunbot"

        terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"]
        self_collisions = 1 
        

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        push_robots = False
  
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
        noise_level = 1 # scales other values
        class noise_scales:
            quat = 0.04
            dof_pos = 0.05
            dof_vel = 0.06
            lin_vel = 0.08
            ang_vel = 0.08
            gravity = 0.05
            height_measurements = 0.1
            pos = 0.06

    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 3.0 
            # close_to_orientation = 2.0 
            to_target = 4
            to_orientation = 2
            stop = 1.0
            lin_vel_limits = -0.001
            # ang_vel_limits = -0.01
            balance = -0.2
            lin_vel_z = -0.01 # 减小高度奖励
            ang_vel_xy = -0.02 # -0.05 # 减小侧摆的奖励
            torques = -0.0000006 # 减小力矩奖励
            # dof_acc = -2.5e-2 # 减小加速度奖励
            action_rate = -0.0002 # 减小动作速率奖励
            overturn = -0.3
            collision = -10.0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        close_para = 15
        tracking_sigma_main = 8 # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 2
        tracking_sigma = 0.5 # tracking reward = exp(-error^2/sigma)
        


class RotunbotTargetObstacleCfgPPO(LeggedRobotCfgPPO):
    seed = 2
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
        experiment_name = 'rotunbot_target_obstacle'
        # num_steps_per_env = 200
        max_iterations = 2500  # number of policy updates
        # resume = True
        # load_run = '/home/an/legged_gym/logs/rotunbot_target/Apr24_00-05-33_' # -1 = last run
        # checkpoint = '1450' # -1 = last saved model
        
