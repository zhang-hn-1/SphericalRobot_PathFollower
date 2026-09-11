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

ball_odom = np.zeros(16)
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
    ball_odom[0] = position.x - 5
    ball_odom[1] = position.y - 5
    ball_odom[2] = position.z +0.4
    ball_odom[3] = orientation.x
    ball_odom[4] = orientation.y
    ball_odom[5] = orientation.z
    ball_odom[6] = orientation.w
    ball_odom[7] = linear_vel.x
    ball_odom[8] = -linear_vel.y
    ball_odom[9] = linear_vel.z
    ball_odom[10] = angular_vel.x
    ball_odom[11] = angular_vel.y
    ball_odom[12] = angular_vel.z
    (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(ball_odom[3:7])
    ball_odom[13] = roll
    ball_odom[14] = pitch
    ball_odom[15] = yaw

def dofstate_callback(msg):
    global dof_state
    first_dof_state = msg.first
    second_dof_state = msg.second
    dof_state[1] = np.pi/18000*first_dof_state.velocity
    dof_state[0] = -second_dof_state.position
    dof_state[2] = -second_dof_state.velocity

def Rotunbot_RL_Control():

    import os
    model=torch.jit.load("policies/policy_nav_5_29_1.pt")
    print(model)
    # obs_input = np.zeros([1, 37], dtype=np.float32)
    obs_input = np.zeros([1, 19], dtype=np.float32)
    action = model(torch.tensor(obs_input))[0].detach().numpy()
    print(action)
    torque = np.zeros(2)
    # print(model(input))
    last_action = np.zeros(2)
    last_ball_odom = np.zeros(16)
    last_torque = np.zeros(2)
    last_dof_state = np.zeros(3)
    command = np.zeros(2)
    decimation = 2
    rospy.init_node("ball_rl_isaac_gym")
    pub = rospy.Publisher('/cmd_vel', Twist,queue_size=10)
    input_pub = rospy.Publisher('/rl_input', Float64MultiArray,queue_size=10)
    output_pub = rospy.Publisher('/rl_output', Float64MultiArray,queue_size=10)
    msg = Twist() #初始化
    command[0]=-2.0
    command[1]=0.0
    rate = rospy.Rate(50)
    control_count = 0
    while not rospy.is_shutdown():
        rospy.Subscriber("/odom", Odometry, odom_callback)
        rospy.Subscriber('/motors_state', motors, dofstate_callback)
        if control_count%decimation ==0:
        
        # obs_input[0 , :2] = command[:]
        # obs_input[0 , 2:15] = ball_odom[:]
        # obs_input[0 , 15:18] = dof_state[:]
        # obs_input[0 , 18:20] = action[:]
        # obs_input[0 , 20:33] = last_ball_odom[:]
        # obs_input[0 , 33:35] = last_action[:]
        # obs_input[0 , 35:37] = last_dof_state[1:3]

            obs_input[0 , :2] = command[:]
            obs_input[0 , 2:5] = ball_odom[:3]
            obs_input[0 , 5:8] = ball_odom[13:16]
            obs_input[0 , 8:14] = ball_odom[7:13]
            obs_input[0 , 14:17] = dof_state[:]
            obs_input[0 , 17:19] = action[:]

            action = model(torch.tensor(obs_input))[0].detach().numpy()
            torque[0] = np.clip(action[0]*1, -3, 3)
            torque[1] = np.clip(action[1]*0.5, -0.52, 0.52)
            # rate_limit_1 = 0.2
            # rate_limit_2 = 0.05
            if torque[0] - last_torque[0]>0.05:
                torque[0] = last_torque[0] + 0.05
            if torque[0] - last_torque[0]<-0.05:
                torque[0] = last_torque[0] - 0.05
            if torque[1] - last_torque[1]>0.01:
                torque[0] = last_torque[0] + 0.01
            if torque[1] - last_torque[1]<-0.01:
                torque[1] = last_torque[1] - 0.01
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
        last_ball_odom[:] = ball_odom[:]
        last_action[:] = action[:]
        last_torque[:] = torque[:]
        last_dof_state[:] = dof_state[:]
        rl_input = Float64MultiArray(data = command)
        rl_output = Float64MultiArray(data = torque)
        input_pub.publish(rl_input)
        output_pub.publish(rl_output)
        print("----------")
        print(ball_odom[0])
        print(ball_odom[1])
        
        pub.publish(msg)
        
        # msg.linear.x=float(-2*0.4)
        # msg.angular.x=float(-0.2)
        # pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    Rotunbot_RL_Control()
