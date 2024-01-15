
import numpy as np
import time

wheel_linear_x = 0
wheel_linear_z = 0
Wheel_angular_z = 0

Kp = 0.01
Ki = 0.0000
Kd = 0.00

P=0.0
I=0.0
D=0.0

x=1.0
y=10.0
Vx=0.0
Vy=10.0

A=0.0
B=0.0
F=0.0
E=10.0

t=1.0
m=10
n=10

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

vector = np.array([[x],[y],[Vx],[Vy]])

predicted_vector = np.array([[0],[0],[0],[0]])

actual_measurements = np.array([[0],[0],[0],[0]])

PID_error = 1
pre_PID_error = 0

image_error = 0

IMU = 0
pre_IMU = 0

F = np.array([[1.0,0.0,t,0.0],[0.0,1.0,0.0,0.0],[B,0.0,A,0.0],[0.0,E,0.0,F]])
H = np.array([[1,0,t,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])

PCM = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
predicted_PCM = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

ProcessNoice_forPCM = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
ProcessNoice_forPredictedVector = np.array([[0],[0],[0],[0]])
measurementNoice = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

end_time = 0


for i in range (m) :
    image_lane = lane[i]
    strt_time = time.time()

    t = (strt_time - end_time)/1000000000

    F[0][2] = t
    H[0][2] = t
    # print(t)

    image_error = 0

    for j in range (n):
        image_error = image_error + image_lane[j]*weight_for_lane/(j+1) #weighted sum from the image data

    # Get IMU readings here 

    print(image_error)

    PID_error = pre_PID_error - (pre_IMU - IMU) + image_error

    print(PID_error)

    # if PID_error < 0 :
    #     PID_error = 0

    P = PID_error
    I = I + PID_error
    D = PID_error - pre_PID_error

    pre_PID_error = PID_error

    F[2,0] = P*Kp + I*Ki + D*Kd
    print(F[2,0])

    # print(vector[2])

    actual_measurements[0] = image_lane[0] ##check
    actual_measurements[1] = vector[1]
    actual_measurements[2] = vector[2]
    actual_measurements[3] = vector[3]


    predicted_vector = np.matmul(F,vector) + ProcessNoice_forPredictedVector
    predicted_PCM = np.matmul(F, np.matmul(PCM,F.transpose())) + ProcessNoice_forPCM

    # print(vector[0])

    S = np.matmul(H,np.matmul(predicted_PCM,H.transpose())) + measurementNoice
    displacement = actual_measurements - np.matmul(H,predicted_vector)

    #kalman gain 
    K = np.matmul(predicted_PCM,np.matmul(H.transpose(),np.linalg.inv(S)))
    
    vector = predicted_vector + np.matmul(K,displacement)
    PCM = np.matmul((np.identity(4)-np.matmul(K,H)),predicted_PCM)

    print(image_lane)
    print(vector.transpose())
    print("ssssssssssssssssssssssssssssssssssssss")

    # vector[0] = image_lane[0]




