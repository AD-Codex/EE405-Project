import cv2
from ultralytics import YOLO

# lane keep variables

import numpy as np
import time

wheel_linear_x = 0
wheel_linear_z = 0
Wheel_angular_z = 0

Kp = 0.01
Ki = 0.00001
Kd = 0.0001

P=0
I=0
D=0

x=1
y=10
Vx=0
Vy=10

A=0
B=0
D=0
E=10

t=1
m=100
n=100

# image_lane = np.array([0,0,0,0,-1,-1,-1,-2,-2,-3])

weight_for_lane = 10   

vector = np.array([[x],[y],[Vx],[Vy]])

predicted_vector = np.array([[0],[0],[0],[0]])

actual_measurements = np.array([[0],[0],[0],[0]])

PID_error = 1
pre_PID_error = 0

image_error = 0

IMU = 0
pre_IMU = 0

F = np.array([[1,0,t,0],[0,1,0,0],[B,0,A,0],[0,E,0,D]])
H = np.array([[1,0,t,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])

PCM = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
predicted_PCM = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

ProcessNoice_forPCM = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
ProcessNoice_forPredictedVector = np.array([[0],[0],[0],[0]])
measurementNoice = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

end_time = 0

# end of lane keep variables

def connect_points_with_lines(points ):

    # # Convert points to NumPy array
    points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    # Draw lines connecting the points
    cv2.polylines(annotated_frame, points_array, isClosed=1, color=(0, 255, 0), thickness=12)



def floor_to_nearest_5(number):
    return 10 * (number // 10)

# (-) for mid point is in the left side  which mean go left
# (+) for mid point is in the right side which mean go right

def merge_dicts(dict1, dict2,pos):
    result_dict = {}

    # Get the intersection of keys from both dictionaries
    common_keys = set(dict1.keys()) & set(dict2.keys())

    # Add common keys and their values to the result dictionary
    for key in common_keys:
        x=(dict1[key] + dict2[key])/2
        cv2.circle(annotated_frame, (int(x),int(key)), 3, (0, 0, 255), thickness=2)
        result_dict[key] = x-pos

    return result_dict

# Load the YOLOv8 model
model = YOLO('lane.pt')

# Open the video file
cap = cv2.VideoCapture(0)

# Get the width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(width,height)



# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    # print(success)

    if success:
        # Run YOLOv8 inference on the frame
        results = model.predict(frame, conf=0.1, classes=[1])
        try:
            annotated_frame = results[0].plot()

            mask=results[0].masks.data.cpu().numpy()[0]

            # Convert annotated_frame to RGB
            # annotated_frame = results[0].masks.data.cpu().numpy()[0]
            # print(results[0].masks.xy[0])
            # print(len(results[0].masks.xy[0]))

            left = dict()
            right=dict()

            line_color=(0, 255, 0)
            v_line_x_pos=int(width/2 -400)
            # Draw the vertical line on the image
            cv2.line(annotated_frame, (v_line_x_pos, 0), (v_line_x_pos, height), line_color, 5)

            points=[]
            img = np.zeros((width, height), dtype=np.uint8)
            for i in range(int(len(results[0].masks.xy[0]))):
                x1, y1 = results[0].masks.xy[0][i]
                x1=int(x1)
                y1=int(y1)
                points.append((x1,y1))
                
                if len(points)>=2:
                    # cv2.line(annotated_frame,points[-1],points[-2],(255,0,0),10)
                    points_array = np.array([points], dtype=np.int32)
                    cv2.polylines(img, [points_array], isClosed=False, color=(255, 0, 0), thickness=10)

                    # Find all non-zero pixels (points) in the image
                    # non_zero_points = np.column_stack(np.where(img > 0))


                    # for point in non_zero_points:
                    #     cv2.circle(annotated_frame, tuple(point), 5, (0, 255, 0), -1)

                    # print(non_zero_points)

                    # img1 = cv2.resize(img, (640, 480))
                    # cv2.imshow("YOLOv8 ",img)

                

                if x1<=v_line_x_pos:
                    left[floor_to_nearest_5(y1)]=x1

                else:
                    right[floor_to_nearest_5(y1)]=x1

                
            # left_list = [left[key] for key in left]
                    
            # connect_points_with_lines(points)

            left=dict(sorted(left.items()))
            right=dict(sorted(right.items()))
            dispacement_dic= merge_dicts(left, right,v_line_x_pos)  
            dispacement_dic=dict(sorted(dispacement_dic.items()))   #  sort from top to bottom
            # dispacement_dic = dict(sorted(dispacement_dic.items(), reverse=True))  #  sort from bottom to top

            #....................................................................for noty...........................
            #  don't change the video, it may not work. trained for few dataset
            # press any key to go to next frame. (select the frames window before press any key)
            #  press q to exit
            #  change waitkey(1) at the end of this code to play as a video 

            #this gives how much displacement values get
            length_of_displacement_list=len(dispacement_dic)

            # following list is the displacement values from top of the screen to bottom. 
            # change the sorting method in line 77 in order to sort from bottom of the screen to top
            displacement_list = [dispacement_dic[key] for key in dispacement_dic]

            # print(length_of_displacement_list)
            # print("displacement ",dispacement_dic)
            # print("displacement list",displacement_list)
            n= len(displacement_list)
            image_lane=displacement_list

            # lane keep algorithm==============================
            # for i in range (m) :

            print(image_lane)

            strt_time = time.time()

            t = (strt_time - end_time)/1000000000

            F[0][2] = t
            H[0][2] = t
            # print(t)

            image_error = 0

            for j in range (n):
                image_error = image_error + image_lane[j]*weight_for_lane/(j+1) #weighted sum from the image data

            # Get IMU readings here 

            # print(image_error)

            PID_error = PID_error - (pre_IMU - IMU) + image_error

            # print(PID_error)

            # if PID_error < 0 :
            #     PID_error = 0

            P = PID_error
            I = I + PID_error
            D = PID_error - pre_PID_error

            F[2,0] = P*Kp + I*Ki + D*Kd
            # print(F[2,0])

            # print(vector[2])

            actual_measurements[0] = image_lane[0] ##check
            actual_measurements[1] = vector[1]
            actual_measurements[2] = vector[2]
            actual_measurements[3] = vector[3]


            predicted_vector = np.matmul(F,vector) + ProcessNoice_forPredictedVector
            predicted_PCM = np.matmul(F, np.matmul(PCM,F.transpose())) + ProcessNoice_forPCM

            # print(vector[0])

            vector = predicted_vector + np.matmul(np.matmul(predicted_PCM, np.matmul(H.transpose(), (np.matmul(H, np.matmul(predicted_PCM, H.transpose())) + measurementNoice))), (actual_measurements - np.matmul(H,predicted_vector)))

            PCM = np.matmul((np.identity(4) - np.matmul(np.matmul(predicted_PCM, np.matmul(H.transpose(), (np.matmul(H, np.matmul(predicted_PCM, H.transpose())) + measurementNoice))), H)), predicted_PCM)

            print(vector.transpose())

            # vector[0] = image_lane[0]

            end_time = time.time()



            # end of lane keep algorithm



            # add any thing here

            # following codes for display in a window
            annotated_frame = cv2.resize(annotated_frame, (640, 480))
            cv2.imshow("YOLOv8 Inference2",annotated_frame)
        except:
            cv2.imshow("YOLOv8 Inference2",frame)

        

        # Break the loop if 'q' is pressed
        if cv2.waitKey(0) & 0xFF == ord("q"): #........................change cv2.waitKey(1) to play video auto
            break
        
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()



