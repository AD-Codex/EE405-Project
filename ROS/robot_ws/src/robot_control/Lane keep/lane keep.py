
import numpy as np

wheel_linear_x = 0
wheel_linear_z = 0
Wheel_angular_z = 0

Kp = 0
Ki = 0
Kd = 0 

P=0
I=0
D=0

x=0
y=10
Vx=0
Vy=10

A=0
B=0
D=0
E=10

t=1
m=10
n=10

image_lane = np.array([10,0,0,0,1,1,1,2,2,3])

weight_for_lane = 10

vector = np.array([[x],[y],[Vx],[Vy]])

predicted_vector = np.array([[0],[0],[0],[0]])

actual_measurements = np.array([[0],[0],[0],[0]])

PID_error = 1000
pre_PID_error = 0

image_error = 0

IMU = 0
pre_IMU = 0

F = np.array([[1,0,t,0],[0,1,0,t],[B,0,A,0],[0,E,0,D]])
H = np.array([[1,0,t,0],[0,1,0,t],[0,0,0,0],[0,0,0,0]])

PCM = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
predicted_PCM = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

ProcessNoice_forPCM = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
ProcessNoice_forPredictedVector = np.array([[0],[0],[0],[0]])
measurementNoice = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])


for i in range (m) :

    image_error = 0

    for j in range (n):
        image_error = image_error + image_lane[j]*weight_for_lane/(j+1) #weighted sum from the image data

    # Get IMU readings here 

    PID_error = PID_error - (pre_IMU - IMU) + image_error

    if PID_error < 0 :
        PID_error = 0

    P = PID_error
    I = I + PID_error
    D = PID_error - pre_PID_error

    F[2,0] = P*Kp + I*Ki + D*Kd

    actual_measurements[0] = image_lane[0]
    actual_measurements[1] = vector[1]
    actual_measurements[2] = vector[2]
    actual_measurements[3] = vector[3]


    predicted_vector = np.matmul(F,vector) + ProcessNoice_forPredictedVector
    predicted_PCM = np.matmul(F, np.matmul(PCM,F.transpose())) + ProcessNoice_forPCM

    vector = predicted_vector + np.matmul(np.matmul(predicted_PCM, np.matmul(H.transpose(), (np.matmul(H, np.matmul(predicted_PCM, H.transpose())) + measurementNoice))), (actual_measurements - np.matmul(H,predicted_vector)))

    PCM = np.matmul((np.identity(4) - np.matmul(np.matmul(predicted_PCM, np.matmul(H.transpose(), (np.matmul(H, np.matmul(predicted_PCM, H.transpose())) + measurementNoice))), H)), predicted_PCM)

    print(vector)

    # vector[0] = image_lane[0]




