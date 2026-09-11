
# Rotunbot Target
    torques = actions.clone().to(self.device)
    actions_scaled[:,0] = torch.clip(actions[:,0] * self.cfg.control.first_actionScale, -6, 6)  #(-8 , 8)
    actions_scaled[:,1] = torch.clip(actions[:,1] * self.cfg.control.second_actionScale, -0.5238, 0.5238)
### 训练参数 5.20:
    termination = -100.0
    close_to_target = 2.0 
    # close_to_orientation = 2.0 
    to_target = 3
    to_orientation = 2
    stop = 1.0
    lin_vel_limits = -0.01
    # ang_vel_limits = -0.01
    balance = -0.05
    lin_vel_z = -0.001 # 减小高度奖励
    ang_vel_xy = -0.002 # -0.05 # 减小侧摆的奖励
    torques = -0.000001 # 减小力矩奖励
    # dof_acc = -2.5e-3 # 减小加速度奖励
    action_rate = -0.2 # 减小动作速率奖励
    overturn = -0.4 

### 训练参数 5.21.1
    termination = -100.0
    # close_to_target = 2.0 
    # close_to_orientation = 2.0 
    to_target = 3
    to_orientation = 2
    stop = 1.0
    lin_vel_limits = -0.5
    # ang_vel_limits = -0.01
    balance = -0.5
    # lin_vel_z = -0.001 # 减小高度奖励
    # ang_vel_xy = -0.002 # -0.05 # 减小侧摆的奖励
    # torques = -0.000001 # 减小力矩奖励
    # dof_acc = -2.5e-3 # 减小加速度奖励
    action_rate = -0.8 # 减小动作速率奖励
    # overturn = -0.4

### 5.25
    class rewards:
        class scales:
            termination = -20.0
            # close_to_target = 1.0 
            # close_to_orientation = 2.0 
            to_target = 3
            to_orientation = 2
            stop = 2.0
            lin_vel_limits = -2.0
            # ang_vel_limits = -0.01
            balance = -0.2
            # lin_vel_z = -0.001 # 减小高度奖励
            ang_vel_xy = -0.02 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            dof_acc = -2.5e-2 # 减小加速度奖励
            action_rate = -1.0 # 减小动作速率奖励
            # overturn = -0.4

### 5.56
    class rewards:
        class scales:
            termination = -20.0
            # close_to_target = 1.0 
            # close_to_orientation = 2.0 
            to_target = 3
            to_orientation = 2
            stop = 2.0
            lin_vel_limits = -20.0
            # ang_vel_limits = -0.01
            balance = -10.0
            # lin_vel_z = -0.001 # 减小高度奖励
            ang_vel_xy = -0.2 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            dof_acc = -2.5e-2 # 减小加速度奖励
            action_rate = -5.0 # 减小动作速率奖励
            # overturn = -0.4

### 5.28
    class control( LeggedRobotCfg.control ):
        control_type = 'R' # P: position, V: velocity, T: torques  R:rotunbot
        # action scale: target torques = actionScale * action
        action_scale = 40
        decimation = 2
        # decimation: Number of control action updates @ sim DT per policy DT
        first_actionScale = 1
        second_actionScale = 0.5
        first_vel_limits = 4
        second_pos_limits = 0.52
        set_a_rate_limit = True
        rate_limit_1 = 2
        rate_limit_2 = 0.3
        torque_limits_1 = 100
        torque_limits_2 = 100
    class rewards:
        class scales:
            termination = -20.0
            # close_to_target = 1.0 
            # close_to_orientation = 2.0 
            to_target = 10
            to_orientation = 0
            stop = 2.0
            lin_vel_limits = -10.0
            # ang_vel_limits = -0.01
            balance = -1.0
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            # dof_acc = -2.5e-3 # 减小加速度奖励
            action_rate = -1.0 # 减小动作速率奖励
            # overturn = -0.4

### 5.29.1
    class control( LeggedRobotCfg.control ):
        control_type = 'R' # P: position, V: velocity, T: torques  R:rotunbot
        # action scale: target torques = actionScale * action
        action_scale = 40
        decimation = 2
        # decimation: Number of control action updates @ sim DT per policy DT
        first_actionScale = 1
        second_actionScale = 0.5
        first_vel_limits = 3
        second_pos_limits = 0.52
        set_a_rate_limit = True
        rate_limit_1 = 0.2
        rate_limit_2 = 0.05
        torque_limits_1 = 100
        torque_limits_2 = 100
    class rewards:
        class scales:
            termination = -20.0
            # close_to_target = 1.0 
            # close_to_orientation = 2.0 
            to_target = 5
            to_orientation = 0
            stop = 1.0
            lin_vel_limits = -10.0
            # ang_vel_limits = -0.01
            balance = -1.0
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            # dof_acc = -2.5e-3 # 减小加速度奖励
            action_rate = -1.0 # 减小动作速率奖励
            # overturn = -0.4

### 5.29.2
    class control( LeggedRobotCfg.control ):
        control_type = 'R' # P: position, V: velocity, T: torques  R:rotunbot
        # action scale: target torques = actionScale * action
        action_scale = 40
        decimation = 2
        # decimation: Number of control action updates @ sim DT per policy DT
        first_actionScale = 1
        second_actionScale = 0.5
        first_vel_limits = 3
        second_pos_limits = 0.52
        set_a_rate_limit = True
        rate_limit_1 = 0.05
        rate_limit_2 = 0.01
        torque_limits_1 = 100
        torque_limits_2 = 100
    class rewards:
        class scales:
            termination = -20.0
            # close_to_target = 1.0 
            # close_to_orientation = 2.0 
            to_target = 5
            to_orientation = 0
            stop = 1.0
            lin_vel_limits = -10.0
            # ang_vel_limits = -0.01
            balance = -10.0
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            # dof_acc = -2.5e-3 # 减小加速度奖励
            action_rate = -5.0 # 减小动作速率奖励
            # overturn = -0.4
    
### 5.30.1
    class control( LeggedRobotCfg.control ):
        control_type = 'R' # P: position, V: velocity, T: torques  R:rotunbot
        # action scale: target torques = actionScale * action
        action_scale = 40
        decimation = 2
        # decimation: Number of control action updates @ sim DT per policy DT
        first_actionScale = 1
        second_actionScale = 0.5
        first_vel_limits = 3
        second_pos_limits = 0.52
        set_a_rate_limit = False
        rate_limit_1 = 0.05
        rate_limit_2 = 0.01
        torque_limits_1 = 100
        torque_limits_2 = 100
    class rewards:
        class scales:
            termination = -40.0
            # close_to_target = 1.0 
            # close_to_orientation = 2.0 
            to_target = 1
            to_orientation = 0
            stop = 1.0
            lin_vel_limits = -10.0
            # ang_vel_limits = -0.01
            balance = -10.0
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            # dof_acc = -2.5e-3 # 减小加速度奖励
            action_rate = -10.0 # 减小动作速率奖励
            # overturn = -0.4
### 6.30
actor_hidden_dims = [256, 64, 32]
critic_hidden_dims = [256, 64, 32]
    class control( LeggedRobotCfg.control ):
        control_type = 'R' # P: position, V: velocity, T: torques  R:rotunbot
        # action scale: target torques = actionScale * action
        action_scale = 40
        decimation = 2
        # decimation: Number of control action updates @ sim DT per policy DT
        first_actionScale = 1
        second_actionScale = 0.5
        first_vel_limits = 3
        second_pos_limits = 0.52
        set_a_rate_limit = True
        rate_limit_1 = 0.05
        rate_limit_2 = 0.01
        torque_limits_1 = 100
        torque_limits_2 = 100
    class rewards:
        class scales:
            termination = -40.0
            close_to_target = 1.0 
            # close_to_orientation = 2.0 
            to_target = 2
            # to_orientation = 2
            stop = 10.0
            lin_vel_limits = -10.0
            # ang_vel_limits = -0.01
            # balance = -10.0
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            # dof_acc = -2.5e-3 # 减小加速度奖励
            action_rate = -0.3 # 减小动作速率奖励
            # overturn = -0.4

### 6.31
    class control( LeggedRobotCfg.control ):
        control_type = 'R' # P: position, V: velocity, T: torques  R:rotunbot
        # action scale: target torques = actionScale * action
        action_scale = 40
        decimation = 2
        # decimation: Number of control action updates @ sim DT per policy DT
        first_actionScale = 1
        second_actionScale = 0.5
        first_vel_limits = 3
        second_pos_limits = 0.52
        set_a_rate_limit = True
        rate_limit_1 = 0.05
        rate_limit_2 = 0.01
        torque_limits_1 = 100
        torque_limits_2 = 100
    class rewards:
        class scales:
            termination = -40.0
            close_to_target = 1.0 
            # close_to_orientation = 2.0 
            to_target = 2
            # to_orientation = 2
            stop = 10.0
            lin_vel_limits = -10.0
            # ang_vel_limits = -0.01
            # balance = -10.0
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            # dof_acc = -2.5e-3 # 减小加速度奖励
            action_rate = -0.003 # 减小动作速率奖励
            # overturn = -0.4
    
### 7.4
    class control( LeggedRobotCfg.control ):
        control_type = 'R' # P: position, V: velocity, T: torques  R:rotunbot
        # action scale: target torques = actionScale * action
        action_scale = 40
        decimation = 2
        # decimation: Number of control action updates @ sim DT per policy DT
        first_actionScale = 1
        second_actionScale = 0.5
        first_vel_limits = 3
        second_pos_limits = 0.52
        set_a_rate_limit = True
        rate_limit_1 = 0.05
        rate_limit_2 = 0.01
        torque_limits_1 = 100
        torque_limits_2 = 100
    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 3.0 
            # close_to_orientation = 2.0 
            to_target = 5
            # to_orientation = 2
            stop = 10.0
            lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            # balance = -10.0
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            # dof_acc = -2.5e-3 # 减小加速度奖励
            action_rate = -0.02 # 减小动作速率奖励
            # overturn = -0.4

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        close_para = 1
        tracking_sigma_main = 5 # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 2
        tracking_sigma = 0.5 # tracking reward = exp(-error^2/sigma)

### 7.4.2
    class control( LeggedRobotCfg.control ):
        control_type = 'R' # P: position, V: velocity, T: torques  R:rotunbot
        # action scale: target torques = actionScale * action
        action_scale = 40
        decimation = 2
        # decimation: Number of control action updates @ sim DT per policy DT
        first_actionScale = 1
        second_actionScale = 0.5
        first_vel_limits = 3
        second_pos_limits = 0.52
        set_a_rate_limit = True
        rate_limit_1 = 0.05
        rate_limit_2 = 0.02
        torque_limits_1 = 100
        torque_limits_2 = 100

    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 3.0 
            # close_to_orientation = 2.0 
            to_target = 5
            # to_orientation = 2
            stop = 50.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            # balance = -10.0
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            # dof_acc = -2.5e-3 # 减小加速度奖励
            action_rate = -0.00002 # 减小动作速率奖励
            # overturn = -0.4

### 7.5
    class control( LeggedRobotCfg.control ):
        control_type = 'R' # P: position, V: velocity, T: torques  R:rotunbot
        # action scale: target torques = actionScale * action
        action_scale = 40
        decimation = 2
        # decimation: Number of control action updates @ sim DT per policy DT
        first_actionScale = 1
        second_actionScale = 0.5
        first_vel_limits = 3
        second_pos_limits = 0.52
        set_a_rate_limit = True
        rate_limit_1 = 0.05
        rate_limit_2 = 0.02
        torque_limits_1 = 100
        torque_limits_2 = 100
    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 2.0 
            # close_to_orientation = 2.0 
            to_target = 5
            # to_orientation = 2
            stop = 50.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            # balance = -10.0
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            # dof_acc = -2.5e-3 # 减小加速度奖励
            action_rate = -0.000005 # 减小动作速率奖励
            # overturn = -0.4

### 7.25
    class control( LeggedRobotCfg.control ):
        control_type = 'R' # P: position, V: velocity, T: torques  R:rotunbot
        # action scale: target torques = actionScale * action
        action_scale = 40
        decimation = 2
        # decimation: Number of control action updates @ sim DT per policy DT
        first_actionScale = 1
        second_actionScale = 0.5
        first_vel_limits = 3
        second_pos_limits = 0.5
        set_a_rate_limit = False
        rate_limit_1 = 0.02
        rate_limit_2 = 0.008
        torque_limits_1 = 100
        torque_limits_2 = 100
    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 0.2 
            # close_to_orientation = 2.0 
            to_target = 1
            # to_orientation = 2
            stop = 5.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            # balance = -0.004
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            # dof_acc = -2.5e-3 # 减小加速度奖励
            action_rate = -0.08 # 减小动作速率奖励
            # overturn = -0.4

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        close_para = 1
        tracking_sigma_main = 2 # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 2
        tracking_sigma = 0.5 # tracking reward = exp(-error^2/sigma)

### 修改了奖励函数7.25
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
    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 0.2 
            # close_to_orientation = 2.0 
            to_target = 1
            # to_orientation = 2
            stop = 200.0
            # arrive = 2.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            # balance = -0.004
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            dof_acc = -2.5e-5 # 减小加速度奖励
            # action_rate = -0.08 # 减小动作速率奖励
            # overturn = -0.4

### 7.28
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
    class commands( LeggedRobotCfg.commands ):
        num_commands = 3 # default: lin_vel_y, ang_vel_yaw
        resampling_time = 30. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        command_yaw = False # if true: compute yaw command from heading error
        stop_distance = 0.1 # (0.1)distance to target before stopping
        class ranges( LeggedRobotCfg.commands.ranges ):
            pos_x = [-5, 5] # min max [m]
            pos_y = [-5, 5] # min max [m]
            yaw = [-3.14, 3.14] # min max [rad]
            lin_vel_x = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 0.2 
            # close_to_orientation = 2.0 
            to_target = 1.5
            # to_orientation = 2
            stop = 200.0
            # arrive = 2.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            # balance = -0.004
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            dof_acc = -2.5e-5 # 减小加速度奖励
            # action_rate = -0.08 # 减小动作速率奖励
            # overturn = -0.4

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        close_para = 1
        tracking_sigma_main = 2 # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 2
        tracking_sigma = 0.5 # tracking reward = exp(-error^2/sigma)

### 7.30
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
        set_a_rate_limit = False
        rate_limit_1 = 0.02
        rate_limit_2 = 0.008
        torque_limits_1 = 100
        torque_limits_2 = 100
    class commands( LeggedRobotCfg.commands ):
        num_commands = 3 # default: lin_vel_y, ang_vel_yaw
        resampling_time = 30. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        command_yaw = True # if true: compute yaw command from heading error
        stop_distance = 0.4 # (0.1)distance to target before stopping
        stop_orientation = 0.4
        class ranges( LeggedRobotCfg.commands.ranges ):
            pos_x = [-5, 5] # min max [m]
            pos_y = [-5, 5] # min max [m]
            yaw = [-3.14, 3.14] # min max [rad]
            lin_vel_x = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 0.4 
            # close_to_orientation = 2.0 
            to_target = 1.5
            # to_orientation = 2
            stop = 20.0
            # arrive = 2.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            # balance = -0.004
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            dof_acc = -2.5e-5 # 减小加速度奖励
            action_rate = -0.0008 # 减小动作速率奖励
            # overturn = -0.4
### 7.31(还行 Jul31_00-01-28_)
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
        set_a_rate_limit = False
        rate_limit_1 = 0.02
        rate_limit_2 = 0.008
        torque_limits_1 = 100
        torque_limits_2 = 100
    class commands( LeggedRobotCfg.commands ):
        num_commands = 3 # default: lin_vel_y, ang_vel_yaw
        resampling_time = 30. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        command_yaw = False # if true: compute yaw command from heading error
        stop_distance = 0.3 # (0.1)distance to target before stopping
        stop_orientation = 0.4
        class ranges( LeggedRobotCfg.commands.ranges ):
            pos_x = [-5, 5] # min max [m]
            pos_y = [-5, 5] # min max [m]
            yaw = [-3.14, 3.14] # min max [rad]
            lin_vel_x = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 0.3 
            # close_to_orientation = 2.0 
            to_target = 4
            # to_orientation = 2
            stop = 100.0
            # arrive = 2.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            # balance = -0.004
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            dof_acc = -2.5e-6 # 减小加速度奖励
            action_rate = -0.00001 # 减小动作速率奖励
            # overturn = -0.4

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        close_para = 1
        tracking_sigma_main = 8 # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 2
        tracking_sigma = 0.5 # tracking reward = exp(-error^2/sigma)
    
### 8.1
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
        set_a_rate_limit = False
        rate_limit_1 = 0.02
        rate_limit_2 = 0.008
        torque_limits_1 = 100
        torque_limits_2 = 100
    class commands( LeggedRobotCfg.commands ):
        num_commands = 3 # default: lin_vel_y, ang_vel_yaw
        resampling_time = 30. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        command_yaw = True # if true: compute yaw command from heading error
        stop_distance = 0.3 # (0.1)distance to target before stopping
        stop_orientation = 0.4
        class ranges( LeggedRobotCfg.commands.ranges ):
            pos_x = [-5, 5] # min max [m]
            pos_y = [-5, 5] # min max [m]
            yaw = [-3.14, 3.14] # min max [rad]
            lin_vel_x = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 0.3 
            # close_to_orientation = 2.0 
            to_target = 4
            # to_orientation = 2
            stop = 80.0
            # arrive = 2.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            # balance = -0.004
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            dof_acc = -2.5e-6 # 减小加速度奖励
            action_rate = -0.00001 # 减小动作速率奖励
            # overturn = -0.4
    
### 8.2 
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
        rate_limit_2 = 0.004
        torque_limits_1 = 100
        torque_limits_2 = 100
    class commands( LeggedRobotCfg.commands ):
        num_commands = 3 # default: lin_vel_y, ang_vel_yaw
        resampling_time = 30. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        command_yaw = False # if true: compute yaw command from heading error
        stop_distance = 0.3 # (0.1)distance to target before stopping
        stop_orientation = 0.4
        class ranges( LeggedRobotCfg.commands.ranges ):
            pos_x = [-5, 5] # min max [m]
            pos_y = [-5, 5] # min max [m]
            yaw = [-3.14, 3.14] # min max [rad]
            lin_vel_x = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 0.3 
            # close_to_orientation = 2.0 
            to_target = 4
            # to_orientation = 2
            stop = 80.0
            # arrive = 2.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            # balance = -0.004
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            dof_acc = -2.5e-5 # 减小加速度奖励
            # action_rate = -0.00001 # 减小动作速率奖励
            # overturn = -0.4

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        close_para = 1
        tracking_sigma_main = 8 # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 2
        tracking_sigma = 0.5 # tracking reward = exp(-error^2/sigma)

### 8.2.2 (在8.2.1基础上)
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
        set_a_rate_limit = False
        rate_limit_1 = 0.02
        rate_limit_2 = 0.004
        torque_limits_1 = 100
        torque_limits_2 = 100

    class commands( LeggedRobotCfg.commands ):
        num_commands = 3 # default: lin_vel_y, ang_vel_yaw
        resampling_time = 30. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        command_yaw = False # if true: compute yaw command from heading error
        stop_distance = 0.3 # (0.1)distance to target before stopping
        stop_orientation = 0.4
        class ranges( LeggedRobotCfg.commands.ranges ):
            pos_x = [-5, 5] # min max [m]
            pos_y = [-5, 5] # min max [m]
            yaw = [-3.14, 3.14] # min max [rad]
            lin_vel_x = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 0.1 
            # close_to_orientation = 2.0 
            to_target = 3
            # to_orientation = 2
            stop = 120.0
            # arrive = 2.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            balance = -0.004
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            dof_acc = -2.5e-5 # 减小加速度奖励
            action_rate = -0.00001 # 减小动作速率奖励
            # overturn = -0.4

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        close_para = 1
        tracking_sigma_main = 8 # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 2
        tracking_sigma = 0.5 # tracking reward = exp(-error^2/sigma)

### 8.2.3 (在8.2.2)
    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 0.1 
            # close_to_orientation = 2.0 
            to_target = 3
            # to_orientation = 2
            stop = 120.0
            # arrive = 2.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            balance = -0.08
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            dof_acc = -2.5e-5 # 减小加速度奖励
            action_rate = -0.0001 # 减小动作速率奖励
            # overturn = -0.4

### 8.4
    class commands( LeggedRobotCfg.commands ):
        num_commands = 3 # default: lin_vel_y, ang_vel_yaw
        resampling_time = 30. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        command_yaw = False # if true: compute yaw command from heading error
        stop_distance = 0.3 # (0.1)distance to target before stopping
        stop_orientation = 0.4
        class ranges( LeggedRobotCfg.commands.ranges ):
            pos_x = [-5, 5] # min max [m]
            pos_y = [-5, 5] # min max [m]
            yaw = [-3.14, 3.14] # min max [rad]
            lin_vel_x = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 0.01 
            # close_to_orientation = 2.0 
            to_target = 1
            # to_orientation = 2
            stop = 120.0
            # arrive = 2.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            balance = -0.8
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            dof_acc = -2.5e-4 # 减小加速度奖励
            action_rate = -0.0005 # 减小动作速率奖励
            # overturn = -0.4

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        close_para = 1
        tracking_sigma_main = 2 # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 2
        tracking_sigma = 0.5 # tracking reward = exp(-error^2/sigma)

### 8.21
    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 0.3 
            # close_to_orientation = 2.0 
            to_target = 4
            # to_orientation = 2
            stop = 80.0
            # arrive = 2.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            balance = -0.08
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            dof_acc = -2.5e-5 # 减小加速度奖励
            action_rate = -0.000005 # 减小动作速率奖励
            # overturn = -0.4

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        close_para = 1
        tracking_sigma_main = 2 # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 2
        tracking_sigma = 0.5 # tracking reward = exp(-error^2/sigma)

### 8.29
    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 1.5 
            # close_to_orientation = 2.0 
            to_target = 1.0
            # to_orientation = 2
            stop = 120.0
            # arrive = 2.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            balance = -5.8
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            dof_acc = -2.5e-4 # 减小加速度奖励
            action_rate = -0.0004 # 减小动作速率奖励
            # overturn = -0.4

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        close_para = 1
        tracking_sigma_main = 8 # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 2
        tracking_sigma = 0.5 # tracking reward = exp(-error^2/sigma)

### 8.29.2
    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 1.5 
            # close_to_orientation = 2.0 
            to_target = 1.0
            # to_orientation = 2
            stop = 150.0
            # arrive = 2.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            balance = -6.8
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            # dof_acc = -2.5e-4 # 减小加速度奖励
            action_rate = -0.00008 # 减小动作速率奖励
            # overturn = -0.4
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rotunbot_target'
        num_steps_per_env = 100
        max_iterations = 3000  # number of policy updates
        resume = True
        load_run = '/home/an/legged_gym/logs/rotunbot_target/Aug02_23-57-43_' # -1 = last run\\Aug02_19-50-29_\\
        checkpoint = '8000' # -1 = last saved model

### 9.2 Sep02_15-10-55_
    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 3.5 
            # close_to_orientation = 2.0 
            to_target = 2.0
            # to_orientation = 2
            stop = 150.0
            # arrive = 2.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            balance = -1.0
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            # dof_acc = -2.5e-4 # 减小加速度奖励
            # action_rate = -0.00006 # 减小动作速率奖励
            # overturn = -0.4
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rotunbot_target'
        num_steps_per_env = 100
        max_iterations = 4000  # number of policy updates
        resume = True
        # load_run = '/home/an/legged_gym/logs/rotunbot_target/Aug29_13-01-29_' # -1 = last run\\Aug02_19-50-29_\\
        # checkpoint = '11000' # -1 = last saved model
        load_run = '/home/an/legged_gym/logs/rotunbot_target/Aug28_14-17-26_' # -1 = last run\\Aug02_19-50-29_\\
        checkpoint = '12000' # -1 = last saved model

### 9.2 
    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 4.5 
            # close_to_orientation = 2.0 
            to_target = 6.0
            # to_orientation = 2
            stop = 80.0
            # arrive = 2.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            balance = -0.5
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            # dof_acc = -2.5e-4 # 减小加速度奖励
            # action_rate = -0.00006 # 减小动作速率奖励
            # overturn = -0.4

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        close_para = 1
        tracking_sigma_main = 8 # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 2
        tracking_sigma = 0.5 # tracking reward = exp(-error^2/sigma)

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rotunbot_target'
        num_steps_per_env = 100
        max_iterations = 6000  # number of policy updates
        resume = True
        # load_run = '/home/an/legged_gym/logs/rotunbot_target/Aug29_13-01-29_' # -1 = last run\\Aug02_19-50-29_\\
        # checkpoint = '11000' # -1 = last saved model
        # load_run = '/home/an/legged_gym/logs/rotunbot_target/Aug28_14-17-26_' # -1 = last run\\Aug02_19-50-29_\\
        # checkpoint = '12000' # -1 = last saved model

        load_run = '/home/an/legged_gym/logs/rotunbot_target/Sep02_15-10-55_' # -1 = last run\\Aug02_19-50-29_\\
        checkpoint = '16000' # -1 = last saved model

### 9.3
    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 4.5 
            # close_to_orientation = 2.0 
            to_target = 6.0
            # to_orientation = 2
            stop = 80.0
            # arrive = 2.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            balance = -0.5
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            # dof_acc = -2.5e-4 # 减小加速度奖励
            # action_rate = -0.00006 # 减小动作速率奖励
            # overturn = -0.4
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rotunbot_target'
        num_steps_per_env = 100
        max_iterations = 3000  # number of policy updates
        # resume = True
        # load_run = '/home/an/legged_gym/logs/rotunbot_target/Aug29_13-01-29_' # -1 = last run\\Aug02_19-50-29_\\
        # checkpoint = '11000' # -1 = last saved model
        # load_run = '/home/an/legged_gym/logs/rotunbot_target/Aug28_14-17-26_' # -1 = last run\\Aug02_19-50-29_\\
        # checkpoint = '12000' # -1 = last saved model

        load_run = '/home/an/legged_gym/logs/rotunbot_target/Sep02_23-10-07_' # -1 = last run\\Aug02_19-50-29_\\
        checkpoint = '22000' # -1 = last saved model

### 9.8
    class control( LeggedRobotCfg.control ):
        control_type = 'R' # P: position, V: velocity, T: torques  R:rotunbot
        # action scale: target torques = actionScale * action
        action_scale = 40
        decimation = 1
        # decimation: Number of control action updates @ sim DT per policy DT
        first_actionScale = 1
        second_actionScale = 0.5
        first_vel_limits = 3  # 1.5
        second_pos_limits = 0.45
        set_a_rate_limit = False
        rate_limit_1 = 0.02
        rate_limit_2 = 0.04
        torque_limits_1 = 100
        torque_limits_2 = 100
    class commands( LeggedRobotCfg.commands ):
        num_commands = 3 # default: lin_vel_y, ang_vel_yaw
        resampling_time = 30. # time before command are changed[s] ##30
        heading_command = False # if true: compute ang vel command from heading error
        command_yaw = False # if true: compute yaw command from heading error
        random_start_yaw = False # if true: robot starts with random heading
        stop_distance = 0.3 # (0.3)distance to target before stopping
        stop_vel = 0.1 # (0.1)velocity to target before stopping
        stop_orientation = 0.4
        class ranges( LeggedRobotCfg.commands.ranges ):
            pos_x = [-5, 5] # min max [m]
            pos_y = [-5, 5] # min max [m]
            yaw = [-3.14, 3.14] # min max [rad]
            lin_vel_x = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 0.9 
            # close_to_orientation = 2.0 
            to_target = 0.5
            # to_orientation = 2
            stop = 20.0
            # arrive = 2.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            balance = -0.8
            time = -1
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            # dof_acc = -2.5e-4 # 减小加速度奖励
            # action_rate = -0.00006 # 减小动作速率奖励
            # overturn = -0.4

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        close_para = 1
        tracking_sigma_main = 8 # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 2

### 9.9
    class control( LeggedRobotCfg.control ):
        control_type = 'R' # P: position, V: velocity, T: torques  R:rotunbot
        # action scale: target torques = actionScale * action
        action_scale = 40
        decimation = 1
        # decimation: Number of control action updates @ sim DT per policy DT
        first_actionScale = 1
        second_actionScale = 0.5
        first_vel_limits = 3  # 1.5
        second_pos_limits = 0.45
        set_a_rate_limit = True
        rate_limit_1 = 0.02
        rate_limit_2 = 0.04
        torque_limits_1 = 100
        torque_limits_2 = 100
    class commands( LeggedRobotCfg.commands ):
        num_commands = 3 # default: lin_vel_y, ang_vel_yaw
        resampling_time = 30. # time before command are changed[s] ##30
        heading_command = False # if true: compute ang vel command from heading error
        command_yaw = False # if true: compute yaw command from heading error
        random_start_yaw = False # if true: robot starts with random heading
        stop_distance = 0.3 # (0.3)distance to target before stopping
        stop_vel = 0.1 # (0.1)velocity to target before stopping
        stop_orientation = 0.4
        class ranges( LeggedRobotCfg.commands.ranges ):
            pos_x = [-5, 5] # min max [m]
            pos_y = [-5, 5] # min max [m]
            yaw = [-3.14, 3.14] # min max [rad]
            lin_vel_x = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
    class rewards:
        class scales:
            termination = -0.0
            close_to_target = 1.2 
            # close_to_orientation = 2.0 
            to_target = 0.5
            # to_orientation = 2
            stop = 20.0
            # arrive = 2.0
            # lin_vel_limits = -1.0
            # ang_vel_limits = -0.01
            balance = -0.8
            time = -1.5
            away_to_target = -0.8
            # lin_vel_z = -0.001 # 减小高度奖励
            # ang_vel_xy = -0.1 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            # dof_acc = -2.5e-4 # 减小加速度奖励
            # action_rate = -0.00006 # 减小动作速率奖励
            # overturn = -0.4

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        close_para = 1
        tracking_sigma_main = 8 # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 2
        tracking_sigma = 0.5 # tracking reward = exp(-error^2/sigma)
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rotunbot_target'
        num_steps_per_env = 100
        max_iterations = 5000  # number of policy updates
        # resume = True
        # load_run = '/home/an/legged_gym/logs/rotunbot_target/Aug29_13-01-29_' # -1 = last run\\Aug02_19-50-29_\\
        # checkpoint = '11000' # -1 = last saved model
        # load_run = '/home/an/legged_gym/logs/rotunbot_target/Aug28_14-17-26_' # -1 = last run\\Aug02_19-50-29_\\
        # checkpoint = '12000' # -1 = last saved model

        # load_run = '/home/an/legged_gym/logs/rotunbot_target/Sep03_11-05-19_' # -1 = last run\\Aug02_19-50-29_\\
        # checkpoint = '25000' # -1 = last saved model
        load_run = '/home/an/legged_gym/logs/rotunbot_target/Sep08_12-19-05_' # -1 = last run\\Aug02_19-50-29_\\
        checkpoint = '10000' # -1 = last saved model

# Rotunbot Vel
### 5.23_1  50hz
    class normalization( LeggedRobotCfg.normalization ):
        class obs_scales:
            command = 5.0
            lin_vel = 1.0
            ang_vel = 0.5
            quat = 1.0
            dof_pos = 2.0
            dof_vel = 1.0
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 3.0
            tracking_ang_vel = 1.5
            # lin_vel_z = -0.002 # 减小高度奖励
            ang_vel_xy = -0.3 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            dof_acc = -2.5e-4 # 减小加速度奖励
            action_rate = -0.5 # 减小动作速率奖励

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        
### 5.23_2  50Hz
    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.5
            tracking_ang_vel = 1.0
            # lin_vel_z = -0.002 # 减小高度奖励
            ang_vel_xy = -0.04 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            dof_acc = -2.5e-2 # 减小加速度奖励
            action_rate = -1.0 # 减小动作速率奖励

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rotunbot_vel'
        num_steps_per_env = 100
        max_iterations = 1000  # number of policy updates

### 5.24_1
    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.5
            tracking_ang_vel = 2.5
            # lin_vel_z = -0.002 # 减小高度奖励
            ang_vel_xy = -0.04 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            dof_acc = -2.5e-2 # 减小加速度奖励
            action_rate = -1.0 # 减小动作速率奖励

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rotunbot_vel'
        num_steps_per_env = 100
        max_iterations = 2000  # number of policy updates

### 8.23
    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.5
            tracking_ang_vel = 2.5
            # lin_vel_z = -0.002 # 减小高度奖励
            ang_vel_xy = -0.04 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            dof_acc = -2.5e-4 # 减小加速度奖励
            action_rate = -0.3 # 减小动作速率奖励

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)

### 8.24
    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.5
            tracking_ang_vel = 2.5
            # lin_vel_z = -0.002 # 减小高度奖励
            ang_vel_xy = -2.0 # -0.05 # 减小侧摆的奖励
            # torques = -0.000001 # 减小力矩奖励
            # dof_acc = -2.5e-4 # 减小加速度奖励
            action_rate = -15 # 减小动作速率奖励

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 2.5 # tracking reward = exp(-error^2/sigma)