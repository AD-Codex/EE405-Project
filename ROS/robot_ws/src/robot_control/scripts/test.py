#!/usr/bin/env python3

import rospy

if __name__ == '__main__':
    rospy.init_node('test_node')

    rospy.loginfo("hello")
    rospy.logwarn("warning")
    rospy.logerr("error")

    rate = rospy.Rate(1)
    i = 0
    while not rospy.is_shutdown():
        text = "Hello " , i
        rospy.loginfo(text)
        rate.sleep()
        i=i+1