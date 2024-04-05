import cv2
from ultralytics import YOLO
import numpy as np
# from linear_regression import *

def perspective_transform(mask_raw,width,height):
    # Convert single channel grayscale to 3 channel image
    mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))

    # Get the size of the original image (height, width, channels)
    h2, w2, c2 = results[0].orig_img.shape
    
    # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
    mask = cv2.resize(mask_3channel, (w2, h2))

    # Define range of brightness in HSV
    lower_black = np.array([0,0,0])
    upper_black = np.array([0,0,1])

    # Create a mask. Threshold the HSV image to get everything black
    mask = cv2.inRange(mask, lower_black, upper_black)

    # Invert the mask to get everything but black
    mask = cv2.bitwise_not(mask)

    # Applying the Canny Edge filter
    edge = cv2.Canny(mask, 50, 150)

    # transposed=perspective(edge)
    height1, width1 = edge.shape
    black_image = np.zeros((height1, width1), dtype=np.uint8)
    
    # Horizontally stack the black image to the frame
    stacked_image = np.hstack((edge, black_image))
    stacked_image = np.hstack((black_image, stacked_image))

    tl=[width1,280]
    bl=[0,height1]
    tr=[2*width1-150,280]
    br=[3*width1,height1]

    # cv2.circle(stacked_image,tl,5,(0,0,255),-1)
    # cv2.circle(stacked_image,bl,5,(0,0,255),-1)
    # cv2.circle(stacked_image,tr,5,(0,0,255),-1)
    # cv2.circle(stacked_image,br,5,(0,0,255),-1)

    pts1=np.float32([tl,bl,tr,br])
    pts2=np.float32([[0,0],[0,height1],[width1,0],[width1,height1]])

    matrix= cv2.getPerspectiveTransform(pts1,pts2)
    transform= cv2.warpPerspective(stacked_image,matrix,(width,height))
    return transform

def get_road_edges(transform):
    indices = np.where(transform != [0])

    # Combine the lists into a list of tuples
    combined_list = list(zip(indices[1],indices[0]))
    combined_list = list(set(combined_list))

    # print("combined_list",combined_list)

    sorted_data = sorted(combined_list, key=lambda x: x[0])
    return sorted_data


# model = YOLO('yolov8n-seg.pt')
model1=YOLO('best.pt')

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("T_junction.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



frame_counter = 0

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    
    
    # height,width=frame.shape[:2]
    # frame=frame[:height,:int(width/2)]
    # Draw the vertical line on the image
    


    if success:
        frame_counter += 1
        # Skip frames if not the 9th frame
        if frame_counter < 6:
            continue

        # Reset the frame counter
        frame_counter = 0
        
        # print("floating_values ", y_values)        
        results = model1.predict(frame, conf=0.1, max_det=1)


        try:
            # Run YOLOv8 inference on the frame
            annotated_frame = results[0].plot(boxes=False)

            if(results[0].masks is not None):
                # Convert mask to single channel image
                mask_raw = results[0].masks[0].cpu().data.numpy().transpose(1, 2, 0)
                
                transform=perspective_transform(mask_raw,width,height)
                height1, width1 = transform.shape
                
                sorted_data=get_road_edges(transform)

                # Convert single-channel frame to three-channel frame
                color_transform_frame = cv2.cvtColor(transform, cv2.COLOR_GRAY2BGR)
                

                # v_line_x_pos=int(width/4)
                v_line_x_pos=int(width1/2)

                # Initialize empty lists for coordinates on the right side and left side
                right_side = []
                left_side = []
                filtered=[]

                # Iterate through the list of coordinates
                for coord in sorted_data:
                    x, y = coord
                    if x <= v_line_x_pos:
                        # Append to the left side list if x <= 100
                        left_side.append((x, y))
                    else:
                        # Append to the right side list if x > 100
                        right_side.append((x, y))

                y_x_max_left = {}  # Dictionary to store the maximum x value for each y value

                # Iterate over the data and update the dictionary
                for x, y in left_side:
                    if y in y_x_max_left:
                        # Update the maximum x value for y if the current x value is greater
                        y_x_max_left[y] = max(y_x_max_left[y], x)
                    else:
                        y_x_max_left[y] = x  # Initialize the maximum x value for y if not already present

                # Convert the dictionary to a list of tuples
                left_side_new = [(x_max, y) for y, x_max in y_x_max_left.items()]


                y_x_min_right = {}

                # Iterate over the data and update the dictionary
                for x, y in right_side:
                    if y in y_x_min_right:
                         y_x_min_right[y] = min(y_x_min_right[y], x)
                    else:
                        y_x_min_right[y] = x

                # Calculate the average x value for each y value and append them to a new list
                right_side_new = [(x_min, y) for y, x_min in y_x_min_right.items()]

                midpoints = []

                # Iterate through all pairs of coordinates in left_side and right_side
                for left_coord in left_side_new:
                    for right_coord in right_side_new:
                        # Check if the y-coordinates match
                        if left_coord[1]== right_coord[1] :
                            # Calculate the average x-value for the same y-coordinate
                            avg_x = int((left_coord[0] + right_coord[0]) / 2)
                            # Append the midpoint coordinates to the midpoints list
                            midpoints.append((avg_x, left_coord[1]))  # or right_coord[1], as they are the same


                # Create a dictionary to store the sum of x values and the count of occurrences for each y value
                y_x_sum_count = {}

                # Iterate over the data and update the dictionary
                for x, y in midpoints:
                    if y in y_x_sum_count:
                        y_x_sum_count[y][0] += x  # Sum of x values
                        y_x_sum_count[y][1] += 1  # Count of occurrences
                    else:
                        y_x_sum_count[y] = [x, 1]

                # Calculate the average x value for each y value and append them to a new list
                averages = [(int(y_x_sum / count), y) for y, (y_x_sum, count) in y_x_sum_count.items()]
            
                # print("Midpoints:", averages)
                for a in averages:
                    x, y = a
                    # OpenCV uses BGR color format, so (0, 0, 255) represents red color
                    if y==250:
                        cv2.circle(color_transform_frame, (x, y), 5, (0, 0, 255), -1)  # -1 specifies to fill the circle
                        cv2.circle(color_transform_frame, (x, y), 70, (0, 0, 255), 0)  # 0 specifies not to fill the circle
                        cv2.circle(color_transform_frame, (x, y), 100, (0, 0, 255), 0)
                        cv2.circle(color_transform_frame, (x, y), 130, (0, 0, 255), 0)
                        cv2.circle(color_transform_frame, (x, y), 160, (0, 0, 255), 0)
                        cv2.circle(color_transform_frame, (x, y), 200, (0, 0, 255), 0)

                        distance_threshold=200

                        junction_count_left=0
                        y_old_left=0
                        bend_threshold=20
                        # left_jun_list=[]

                        for b in left_side_new:
                            x3,y3 =b

                            if y3<=251+distance_threshold and y3>=249-distance_threshold:
                                # cv2.circle(color_transform_frame, (x3, y3), 2, (255, 0, 255), -1)
                                distance = int(np.sqrt((x - x3)**2 + (y - y3)**2))
                                # print("distance ",distance)
                                if distance==distance_threshold:
                                    if abs(y_old_left - y3) > bend_threshold or y_old_left == 0:
                                        y_old_left=y3
                                        junction_count_left+=1
                                        cv2.circle(color_transform_frame, (x3, y3), 5, (0, 255, 255), -1)
                                        # print("left x3,y3",x3,y3)

                        junction_count_right=0
                        y_old_right=0

                        for b in right_side_new:
                            x3,y3 =b

                            if y3<=251+distance_threshold and y3>=249-distance_threshold:
                                # cv2.circle(color_transform_frame, (x3, y3), 2, (255, 0, 255), -1)
                                distance = int(np.sqrt((x - x3)**2 + (y - y3)**2))
                                # print("distance ",distance)
                                if distance==distance_threshold:
                                    if abs(y_old_right - y3) > bend_threshold or y_old_right == 0:
                                        y_old_right=y3
                                        junction_count_right+=1
                                        cv2.circle(color_transform_frame, (x3, y3), 5, (0, 255, 255), -1)
                                        # print("right x3,y3",x3,y3)


                    else:
                        cv2.circle(color_transform_frame, (x, y), 2, (255, 0, 0), -1)  # -1 specifies to fill the circle
                    # cv2.circle(annotated_frame, (x, y), 5, (255, 255, 255), -1) 

                # print("junction left", junction_count_left)
                # print("junction right", junction_count_right)

                if junction_count_left>2 and junction_count_left!=0:
                    image = cv2.putText(color_transform_frame, 'left Turn', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 150), 1, cv2.LINE_AA)
                elif junction_count_right>2 and junction_count_right!=0:
                    image = cv2.putText(color_transform_frame, 'Right Turn', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 150), 1, cv2.LINE_AA)
              


                cv2.line(color_transform_frame, (v_line_x_pos, 0), (v_line_x_pos, height), (0,255,0), 2)

                displacement_dict = {y: x - v_line_x_pos for x, y in midpoints}
                displacement_dict = dict(sorted(displacement_dict.items(), key=lambda item: item[0], reverse=True))
                # Generate 25 equally spaced floating-point values between 0 and 800
                floating_values = np.linspace(0, height1, 10)
                y_values = floating_values.astype(int)
                # Convert floating values to a set for faster lookup
                floating_values_set = set(y_values)

                # Filter the dictionary based on the floating values
                filtered_dict = {key: value for key, value in displacement_dict.items() if key in floating_values_set}
                cord_list = [(value+v_line_x_pos, key) for key, value in filtered_dict.items()]

                displacement_list = list(filtered_dict.values())
                print("list", displacement_list)
                print("coordinates", cord_list)

                # for a in cord_list:
                #     x,y=a
                #     cv2.circle(color_transform_frame, (x, y), 5, (100, 20, 25), -1)

                
                n= len(displacement_list)
                image_lane=displacement_list

                # print(image_lane)
                

                

                # Show the masked part of the image
                cv2.imshow("mask", mask_raw)
                resized_mask = cv2.resize(mask_raw, (int(mask_raw.shape[1]/4), int(mask_raw.shape[0]/4)))
                # print("mask shape",resized_mask.shape)
                cv2.imshow("transpose",color_transform_frame)

            cv2.imshow("YOLOv8 Inference2",annotated_frame)


            

            # object detection
            # results0 = model.predict(annotated_frame, conf=0.25, classes=[0,1,2,3,5,7,9,11,15,16])
            # try:
            #     annotated_frame1 = results0[0].plot(boxes=False)

            #     annotated_frame1 = cv2.resize(annotated_frame1, (640, 480))
            #     cv2.imshow("YOLOv8 Inference2",annotated_frame1)
            # except:
            #     print('no object')
            #     cv2.imshow("YOLOv8 Inference2",annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
        except:
            # cv2.imshow("YOLOv8 Inference2",frame)
            print("except part is running")
            cv2.imshow("YOLOv8 Inference2",frame)



    # Break the loop if 'q' is pressed
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break



  
# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()



