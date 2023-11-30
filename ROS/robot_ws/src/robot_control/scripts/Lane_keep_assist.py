


import rospy
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

wheel_linear_x = 0
wheel_linear_z = 0
Wheel_angular_z = 0

x=0
y=0
Vx=0
Vy=10

A=0
B=0
D=0
E=0

t=1

vector = np.array([[x],[y],[Vx],[Vy]])

predicted_vector = np.array([[0],[0],[0],[0]])

F = np.array([[1,0,t,0],[0,1,0,t],[B,0,A,0],[0,E,0,D]])
H = np.array([[1,0,t,0],[0,1,0,t],[0,0,0,0],[0,0,0,0]])

PCM = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
predicted_PCM = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])


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
        #msg_write = Twist()
        #msg_write.linear.x = wheel_linear_x*100
        #msg_write.angular.z = Wheel_angular_z*100

        #msg_write.linear.z = wheel_linear_z


        



        pub.publish(msg_write)
        rate.sleep()

    rospy.spin()