#include <Arduino.h>
#include <ros.h>
#include "move.h"
#include <std_msgs/String.h>
#include <geometry_msgs/Twist.h>

int move_linear_x = 0;
int move_angular_z = 0;
int move_state = 0;
int fn_return = 0;


void onTwist(const geometry_msgs::Twist &msg);

ros::NodeHandle  nh;

std_msgs::String str_msg_write;
ros::Publisher chatter("4_wheel/write", &str_msg_write);

ros::Subscriber<geometry_msgs::Twist> sub("4_wheel/cmd_vel", &onTwist);


void setup() {
  move_pin_init();
  
  nh.initNode();
  nh.advertise(chatter);
  nh.subscribe(sub); 
}

void loop() {
  
  nh.spinOnce();
  
  Wheel_move(move_state, move_linear_x, move_angular_z);
  delay(10);
  
}


void onTwist(const geometry_msgs::Twist &msg) {
  move_linear_x = msg.linear.x;
  move_state = msg.linear.z;
  move_angular_z = msg.angular.z;

  char read_data[40];
  sprintf(read_data, "state:%d linear_x:%d anguler_z:%d", move_state, move_linear_x, move_angular_z);
  ros_log(read_data);  
}

void ros_log(char* msg) {
  str_msg_write.data = msg;
  chatter.publish( &str_msg_write);
}
