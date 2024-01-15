
import numpy as np
import time
import math

class PIDController:
    def __init__(self, kp, ki, kd, target):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.prev_error = 0
        self.integral = 0

    def update(self, current_value, dt):
        error = self.target - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.prev_error = error
        return output


Kp_speed = 0.01
Ki_speed = 0.0000
Kd_speed = 0.00

Kp_angular = 0.01
Ki_angular = 0.0000
Kd_angular = 0.00

Kp_displacement = 0.01
Ki_displacement = 0.0000
Kd_displacement = 0.00

x=1.0
speed=10.0
thita=0.0
angular_velocity=10.0

A=0.01
B=0.01
J=0.0
E=10.0

t=1.0
m=10
n=10

#These are the input data to the controller 
given_speed = 100
given_angle = 0
given_imu = 0


#test lane data
lane = np.array([[0,0,0,0,1,1,1,2,2,3],
                [0,0,0,1,1,1,1,2,2,4],
                [1,1,1,1,1,1,1,2,2,3],
                [0,0,0,0,1,1,1,2,2,3],
                [-1,-1,-1,-1,0,0,0,1,1,1],
                [-2,-2,-2,-2,-1,-1,-1,0,0,1],
                [-3,-3,-3,-1,0,0,0,1,1,1],
                [-4,-4,-3,-1,0,0,0,1,1,1],
                [0,0,0,0,1,1,1,2,2,3],
                [0,0,0,0,1,1,1,2,2,3]])

weight_for_lane = 10   
vector = np.array([[x],[speed],[thita],[angular_velocity]])

predicted_vector = np.array([[0],[0],[0],[0]])

actual_measurements = np.array([[0],[0],[0],[0]])

PID_error = 1
pre_PID_error = 0

image_error = 0

IMU = 0
pre_IMU = 0
road_gradient = 0 #get this from map data or camera

road_gradient = pre_IMU # only when GPS not available 

given_imu = road_gradient - IMU

speed_out = 0
angular_out = 0
displacement_out = 0

#PID ===============
pid_speed = PIDController(Kp_speed, Ki_speed, Kd_speed, given_speed)
pid_angular = PIDController(Kp_angular, Ki_angular, Kd_angular, given_angle)
pid_displacement = PIDController(Kp_displacement, Ki_displacement, Kd_displacement, image_error)

F = np.array([[1.0,t*math.sin(thita),0.0,0.0],[0.0,0.0,A,B],[0.0,0.0,1,t],[0.0,0.0,0.0,0.0]])
H = np.array([[1.0,t*math.sin(thita),0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,1,t],[0.0,0.0,0.0,0.0]])

PCM = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
predicted_PCM = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

ProcessNoice_forPCM = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
ProcessNoice_forPredictedVector = np.array([[0],[0],[0],[0]])
measurementNoice = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
given_data = np.array([[J],[speed_out],[given_imu],[angular_out-displacement_out]])

end_time = 0



for i in range (m) :

    dt = 0.1

    image_lane = lane[i]
    strt_time = time.time()

    J = image_lane[0]
    given_data[0] = J

    t = (strt_time - end_time)/1000000000

    F[0][2] = t
    H[0][2] = t
    # print(t)

    image_error = 0

    for j in range (n):
        image_error = image_error + image_lane[j]*weight_for_lane/(j+1) #weighted sum from the image data

    
    #PID given data update
    # pid_speed.update(given_speed)
    # pid_angular.update_target(given_angle)
    # pid_displacement.update_target(image_error)

    #PID outputs 
    pid_speed_output = pid_speed.update(speed_out, dt)
    pid_angular_output = pid_angular.update(angular_out, dt)
    pid_displacement_output = pid_displacement.update(displacement_out, dt)

    speed_out += pid_speed_output   
    angular_out += pid_angular_output
    displacement_out += pid_displacement_output

    given_data = np.array([[J],[speed_out],[given_imu],[angular_out-displacement_out]])
    F = np.array([[1.0,t*math.sin(vector[2]),0.0,0.0],[0.0,0.0,A,B],[0.0,0.0,1,t],[0.0,0.0,0.0,0.0]])
    H = np.array([[1.0,t*math.sin(vector[2]),0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,1,t],[0.0,0.0,0.0,0.0]])

    print(image_error)


    print(PID_error)

    road_gradient = pre_IMU # only when GPS not available 

    given_imu = road_gradient - IMU


    actual_measurements[0] = image_lane[0] ##check
    actual_measurements[1] = vector[1]
    actual_measurements[2] = given_imu
    actual_measurements[3] = vector[3]


    predicted_vector = np.matmul(F,vector) + ProcessNoice_forPredictedVector + given_data
    predicted_PCM = np.matmul(F, np.matmul(PCM,F.transpose())) + ProcessNoice_forPCM

    # print(vector[0])

    S = np.matmul(H,np.matmul(predicted_PCM,H.transpose())) + measurementNoice
    displacement = actual_measurements - np.matmul(H,predicted_vector)

    print(S)
    print(H)
    print(predicted_PCM)
    print(F)

    #kalman gain 
    K = np.matmul(predicted_PCM,np.matmul(H.transpose(),np.linalg.inv(S)))
    
    vector = predicted_vector + np.matmul(K,displacement)
    PCM = np.matmul((np.identity(4)-np.matmul(K,H)),predicted_PCM)

    pre_IMU = IMU

    print(image_lane)
    print(vector.transpose())
    print("ssssssssssssssssssssssssssssssssssssss")

    # vector[0] = image_lane[0]




