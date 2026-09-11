from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class RotunbotRoughCfg( LeggedRobotCfg ):
    """
    Configuration class for the Rotunbot spherical robot.
    """
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_actions = 2
        frame_stack = 20      #all histroy obs num
        short_frame_stack = 5   #short history step
        c_frame_stack = 3  #all histroy privileged obs num
        num_single_obs = 14
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 21
        single_linvel_index = 8
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 2
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
        }

    class control():
        control_type = 'T' # P: position, V: velocity, T: torques, R:Rotunbot
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target torques = actionScale * action
        action_scale = 10
        first_actionScale = 10.0
        second_actionScale = 0.5236
        torque_limits =50
        torque_limits_1 = 50
        torque_limits_2 = 50
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class commands( LeggedRobotCfg.commands ):
        num_commands = 2 # default: lin_vel_y, ang_vel_yaw
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_y = [-2.0, 2.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
    
    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/Rotunbot/urdf/Rotunbot.urdf"
        name = "Rotunbot"

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
        noise_level = 1.0 # scales other values
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
            tracking_ang_vel = 1.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            torques = -0.00004
            dof_acc = -2.5e-7
            action_rate = -0.02
            dof_pos_limits = -5

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    
    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.
        


class RotunbotRoughCfgPPO(LeggedRobotCfgPPO):

    seed = 1
    runner_class_name = 'DWLOnPolicyRunner'
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        state_estimator_hidden_dims=[256, 128, 64]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        kernel_size=[3, 2]
        filter_size=[16, 8]
        stride_size=[1, 1]
        lh_output_dim=16   #long history output dim
        in_channels = RotunbotRoughCfg.env.frame_stack
        
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        lin_vel_idx = RotunbotRoughCfg.env.single_num_privileged_obs * (RotunbotRoughCfg.env.c_frame_stack - 1) + RotunbotRoughCfg.env.single_linvel_index

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCriticDWL'
        algorithm_class_name = 'PPODWL'
        run_name = ''
        experiment_name = 'rotunbot'
        max_iterations = 800  # number of policy updates
        
