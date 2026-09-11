import torch
import numpy as np
import rospy
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray ,Float64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from communication_msgs.msg import motors
from geometry_msgs.msg import Twist
from communication_msgs.msg import motor
import tf

ball_odom = np.zeros(13)
dof_state = np.zeros(3)
last_dof_state = np.zeros(4)
last_last_dof_state = np.zeros(4)
ball_euler = np.zeros(3)

def odom_callback(msg):
    global ball_odom
    # 获取位置和姿态
    position = msg.pose.pose.position
    orientation = msg.pose.pose.orientation
    linear_vel = msg.twist.twist.linear
    angular_vel = msg.twist.twist.angular
    # 将新的数据添加到DataFrame
    ball_odom[0] = orientation.x
    ball_odom[1] = orientation.y
    ball_odom[2] = orientation.z
    ball_odom[3] = orientation.w
    ball_odom[4] = linear_vel.x
    ball_odom[5] = -linear_vel.y
    ball_odom[6] = linear_vel.z
    ball_odom[7] = angular_vel.x
    ball_odom[8] = angular_vel.y
    ball_odom[9] = angular_vel.z
    (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(ball_odom[:4])
    ball_odom[10] = roll
    ball_odom[11] = pitch
    ball_odom[12] = yaw

def ball_roll_callback(msg):
    global ball_euler
    ball_euler[0] = -msg.data

def ball_pitch_callback(msg):
    global ball_euler
    ball_euler[1] = msg.data

def ball_yaw_callback(msg):
    global ball_euler
    ball_euler[2] = msg.data

def dofstate_callback(msg):
    global dof_state
    first_dof_state = msg.first
    second_dof_state = msg.second
    dof_state[1] = np.pi/18000*first_dof_state.velocity
    dof_state[0] = -second_dof_state.position
    dof_state[2] = -second_dof_state.velocity

def get_action_sequence1(dt, episode_length_buf):
    # 将时间计数转换为秒
    time = episode_length_buf * dt  # self.dt = 0.02
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
        if time < 1.0:
            action1 = 0.0
        elif 1.0 <= time < 8.0:
            # 第 2 秒到第 3 秒，线性上升
            action1 = -3.0
        else:
            action1 = 0.0

        # 动作序列逻辑
        if time < 2.0:
            action2 = 0.0
        elif 2.0 <= time < 8.0:
            action2 = 0.4 * np.sin(2 * np.pi * (time - 2.0) / 6.0)
        else:
            action2 = 0.0

        # 生成第二个动作
        
        return [action1,action2]

def Rotunbot_action_list():
    """
    生成一个动作列表，包含每个时间步的动作值。
    返回:
        list: 包含每个时间步的动作值的列表。
    """
    torque = np.zeros(2)
    rospy.init_node("action_sequence")
    pub = rospy.Publisher('/cmd_vel', Twist,queue_size=10)
    output_pub = rospy.Publisher('/rl_output', Float64MultiArray,queue_size=10)
    msg = Twist() #初始化
    # command = [-0.6 , -0.2]
    
    rate = rospy.Rate(50)
    control_count = 0
    dt = 0.02  # 每个时间步的间隔
    total_time = 10.0  # 总时间 20 秒
    max_episode_length = total_time / dt  # 最大时间步数
    episode_length_bufs = np.arange(0, total_time / dt)  # 时间计数
    while not rospy.is_shutdown():
        torque = get_action_sequence1(dt, control_count)
        msg.linear.x=float(torque[0]*0.4)
        msg.angular.x=float(-torque[1])
        pub.publish(msg)
        control_count +=1
        # msg.linear.x=float(1*0.4)
        # msg.angular.x=float(-0.2)
        # pub.publish(msg)
        rate.sleep()

    
    
def Rotunbot_RL_Control():

    import os
    # model=torch.jit.load("policies/policy_1.pt")
    model=torch.jit.load("policies/policy_5_25.pt")#24_1,2
    model1=torch.jit.load("policies/policy_4.pt")
    print(model)
    obs_input = np.zeros([1, 16], dtype=np.float32)
    obs_input1 = np.zeros([1, 26], dtype=np.float32)
    obs_input2 = np.zeros([1, 13], dtype=np.float32)
    obs_input3 = np.zeros([1, 16], dtype=np.float32)
    # obs_input4 = np.zeros([1, 9], dtype=np.float32)
    action = model(torch.tensor(obs_input4))[0].detach().numpy()
    print(action)
    torque = np.zeros(2)
    # print(model(input))
    last_action = np.zeros(2)
    last_last_action = np.zeros(2)
    last_ball_odom = np.zeros(13)
    last_action = np.zeros(2)
    last_dof_state = np.zeros(3)
    command = np.zeros(2)
    rospy.init_node("ball_rl_isaac_gym")
    pub = rospy.Publisher('/cmd_vel', Twist,queue_size=10)
    output_pub = rospy.Publisher('/rl_output', Float64MultiArray,queue_size=10)
    msg = Twist() #初始化
    # command = [-0.6 , -0.2]
    command[0]=-0.3
    command[1]=0.1
    rate = rospy.Rate(50)
    decimation = 2
    control_count = 0
    while not rospy.is_shutdown():
        rospy.Subscriber("/odom", Odometry, odom_callback)
        rospy.Subscriber('/motors_state', motors, dofstate_callback)
        if control_count%decimation ==0:
            yaw_vel = 25*(ball_odom[12] - last_ball_odom[12])
            
            obs_input2[0 , :2] =command[:] * 5
            obs_input2[0 , 2:5] = ball_odom[10:13] * 1
            obs_input2[0 , 5:8] = ball_odom[4:7] * 1
            obs_input2[0 , 8:11] = ball_odom[7:10] * 0.5
            obs_input2[0 , 11:13] = last_action[:]
            # obs_input4[0 , 5:11] = ball_odom[4:10]
            # obs_input4[0 , 11:14] = ball_odom[7:10]
            # obs_input4[0 , 14:17] = ball_odom[7:10]
            obs_input2 = np.clip(obs_input2,-100,100)

            action = model(torch.tensor(obs_input2))[0].detach().numpy()
            action = np.clip(action,-100,100)
            # torque[0] = np.clip(action[0]*1, -8, 8)
            # torque[1] = np.clip(action[1]*0.5, -0.5236, 0.5236)
            torque[0] = np.clip(action[0]*1, -6, 6)
            torque[1] = np.clip(action[1]*0.5, -0.5, 0.5)
            last_ball_odom[:] = ball_odom[:]
        # # if torque[0]>=0:
        # #     msg.linear.x=float((torque[0]*8 + 80)/100)
        # # else:
        # #     msg.linear.x=float((torque[0]*8 - 80)/100)
        
        # # if torque[1]>=0:
        # #     msg.angular.x=float((torque[1]*10 + 60)/18000*3.1415926)
        # # else:
        # #     msg.angular.x=float((torque[1]*10 - 60)/18000*3.1415926)
        msg.linear.x=float(torque[0]*0.4)
        msg.angular.x=float(-torque[1])
        
        last_action[:] = action[:]
        last_dof_state[:] = dof_state[:]
        rl_output = Float64MultiArray(data = torque)
        output_pub.publish(rl_output)
        print("----------")
        print(ball_odom[4])
        print(yaw_vel)
        
        pub.publish(msg)
        control_count +=1
        # msg.linear.x=float(1*0.4)
        # msg.angular.x=float(-0.2)
        # pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    Rotunbot_action_list()
    # Rotunbot_RL_Control()
