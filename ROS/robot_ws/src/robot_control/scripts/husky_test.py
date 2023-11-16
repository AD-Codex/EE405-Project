#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Imu


husky_linear_x = 0
husky_angular_z = 0

# imu callbavk fn
def imu_callback(msg : Imu):
    global husky_linear_x
    global husky_angular_z

    msg_write = Twist()
    msg_write.linear.x = husky_linear_x
    msg_write.angular.z = husky_angular_z
    husky_pub.publish(msg_write)


# joystick callback fn
def joy_callback(msg : Joy):
    global husky_linear_x
    global husky_angular_z

    if (msg.axes[1]== 0):
        husky_linear_x = 0
    else:
        husky_linear_x = msg.axes[1]

    if (msg.axes[3]== 0):
        husky_angular_z = 0
    else:
        husky_angular_z = 2*msg.axes[3]  


    if (msg.buttons[4] == 1 or msg.buttons[6] == 1):
        husky_linear_x = 0.0
        husky_angular_z = 0.0


if __name__ == '__main__':
    rospy.init_node('husky_node')
    rospy.loginfo("husky node start")

    husky_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    sub = rospy.Subscriber("/imu/data", Imu, callback=imu_callback)
    Joy_sub = rospy.Subscriber("/joy", Joy, callback=joy_callback)

    rospy.spin()

   
