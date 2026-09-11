from legged_gym.envs import RotunbotRoughCfg, RotunbotRoughCfgPPO

class AnymalCRoughCfgPPO( RotunbotRoughCfgPPO ):
    class runner( RotunbotRoughCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_anymal_c'
        load_run = -1


class AnymalCFlatCfg( RotunbotRoughCfg ):
    class env( RotunbotRoughCfg.env ):
        num_observations = 33
  
    class terrain( RotunbotRoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
  
    class asset( RotunbotRoughCfg.asset ):
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

    class rewards( RotunbotRoughCfg.rewards ):
        max_contact_force = 350.
        class scales ( RotunbotRoughCfg.rewards.scales ):
            orientation = -5.0
            torques = -0.000025
            feet_air_time = 2.
            # feet_contact_forces = -0.01
    
    class commands( RotunbotRoughCfg.commands ):
        heading_command = False
        resampling_time = 4.
        class ranges( RotunbotRoughCfg.commands.ranges ):
            ang_vel_yaw = [-1.5, 1.5]

    class domain_rand( RotunbotRoughCfg.domain_rand ):
        friction_range = [0., 1.5] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.

#PPO训练设置
class AnymalCFlatCfgPPO( AnymalCRoughCfgPPO ):
    class policy( AnymalCRoughCfgPPO.policy ):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( AnymalCRoughCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner ( AnymalCRoughCfgPPO.runner):
        run_name = ''
        experiment_name = 'flat_anymal_c'
        load_run = -1
        max_iterations = 300
