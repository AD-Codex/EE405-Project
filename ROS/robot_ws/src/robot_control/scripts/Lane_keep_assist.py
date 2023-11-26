


import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

wheel_linear_x = 0
wheel_linear_z = 0
Wheel_angular_z = 0

# joystick callback fn
def joy_callback(msg : Joy):
    global wheel_linear_x
    global wheel_linear_z
    global Wheel_angular_z

    wheel_linear_x = msg.axes[1]
    Wheel_angular_z = msg.axes[3]

    if ( msg.buttons[4] == 1 or msg.buttons[6] == 1) :
        wheel_linear_z = 1
    else :
        wheel_linear_z = 0



if __name__ == '__main__':
    rospy.init_node('4Wheel_node')
    rospy.loginfo("4Wheel node start")

    pub = rospy.Publisher("/4_wheel/cmd_vel", Twist, queue_size=10)
    Joy_sub = rospy.Subscriber("/joy", Joy, callback=joy_callback)

    rate = rospy.Rate(20)

    while not rospy.is_shutdown():
        msg_write = Twist()
        msg_write.linear.x = wheel_linear_x*100
        msg_write.angular.z = Wheel_angular_z*100

        msg_write.linear.z = wheel_linear_z

        pub.publish(msg_write)
        rate.sleep()

    rospy.spin()