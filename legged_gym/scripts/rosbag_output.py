# coding:utf-8
#!/usr/bin/python
 
# Extract images from a bag file.
import os
import sys
import roslib
import rosbag
import rospy
import tf

odom_Data = open("data/odom_data_raw.txt", 'w')
motor_Data = open("data/motor_data_raw.txt", 'w')
policy_Data = open("data/policy_data_raw.txt", 'w')
state_Data = open("data/state_data_raw.txt", 'w')


class OutputCreator():
    def __init__(self):
        with rosbag.Bag('/home/an/data/7_30/2025-07-29-21-02-30.bag', 'r') as bag:  # 要读取的bag文件；
            state_Data.writelines(["time", " ", str("px"), " ", str("py"), " ", "roll", " ", "pitch", " ", "yaw", " ", "vx",
                        " ", "vy", " ", "vz", " ", "wx", " ", str("wy"), " ", str("wz")," ", str("first_vel"), " ", str("first_cur"), " ", 
                     str("second_vel"), " ", str("second_pos"), " ", str("second_cur")," ", str("first")," ", str("second"), "\r\n"])
            px = 0
            py = 0
            roll = 0
            pitch = 0
            yaw = 0
            vx = 0
            vy = 0
            vz = 0
            wx = 0
            wy = 0
            wz = 0
            first = 0
            second = 0
            first_vel = 0
            first_cur = 0
            second_vel = 0
            second_pos = 0
            second_cur = 0
            for topic, msg, t in bag.read_messages():
                if topic == "/odom":  # msf topic
                    timestr = "%2.2f" % msg.header.stamp.to_sec()
                    px = msg.pose.pose.position.x
                    py = msg.pose.pose.position.y
                    pz = msg.pose.pose.position.z
                    qx = msg.pose.pose.orientation.x
                    qy = msg.pose.pose.orientation.y
                    qz = msg.pose.pose.orientation.z
                    qw = msg.pose.pose.orientation.w
                    (roll, pitch, yaw) = tf.transformations.euler_from_quaternion([qx, qy, qz, qw])
                    
                    vx = msg.twist.twist.linear.x
                    vy = msg.twist.twist.linear.y
                    vz = msg.twist.twist.linear.z
                    wx = msg.twist.twist.angular.x
                    wy = msg.twist.twist.angular.y
                    wz = msg.twist.twist.angular.z
                    odom_Data.writelines([timestr, " ", str(px), " ", str(py), " ", str(
                        pz), " ", str(roll), " ", str(pitch), " ", str(yaw), " ", str(vx),
                        " ", str(vy), " ", str(vz), " ", str(wx), " ", str(wy), " ", str(wz), "\r\n"])
                    state_Data.writelines([timestr, " ", str(px), " ", str(py), " ", str(roll), " ", str(pitch), " ", str(yaw), " ", str(vx),
                        " ", str(vy), " ", str(vz), " ", str(wx), " ", str(wy), " ", str(wz)," ", str(first_vel), " ", str(first_cur), " ", 
                     str(second_vel), " ", str(second_pos), " ", str(second_cur)," ", str(first)," ", str(second), "\r\n"])
                    
                elif topic == "/motors_state":  # msf topic
                    timestr = "%.2f" % msg.header.stamp.to_sec()
                    first_vel = float(msg.first.velocity / 18000 * 3.1416)
                    first_cur = msg.first.current
                    second_vel = msg.second.velocity
                    second_pos = msg.second.position
                    second_cur = msg.second.current
                    motor_Data.writelines([timestr, " ", str(first_vel), " ", str(first_cur), " ", 
                    str(second_vel), " ", str(second_pos), " ", str(second_cur), "\r\n"])
                    

                elif topic == "/cmd_vel":  # msf topic
                    
                    first = msg.linear.x
                    second = msg.angular.x
                    policy_Data.writelines([timestr, " ", str(first)," ", str(second), "\r\n"])
                    # state_Data.writelines([timestr, " ", str(px), " ", str(py), " ", str(
                    #     pz), " ", str(roll), " ", str(pitch), " ", str(yaw), " ", str(vx),
                    #     " ", str(vy), " ", str(vz), " ", str(wx), " ", str(wy), " ", str(wz)," ", str(first_vel), " ", str(first_cur), " ", 
                    # str(second_vel), " ", str(second_pos), " ", str(second_cur)," ", str(first)," ", str(second), "\r\n"])


if __name__ == '__main__':
    try:
        output_creator = OutputCreator()
    except rospy.ROSInterruptException:
        pass