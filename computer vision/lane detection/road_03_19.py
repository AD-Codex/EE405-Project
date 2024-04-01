import cv2
from ultralytics import YOLO
import numpy as np
from linear_regression import *


# model = YOLO('yolov8n-seg.pt')
model1=YOLO('best.pt')

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("2.mp4")
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
                
                # Convert single channel grayscale to 3 channel image
                mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))

                # Get the size of the original image (height, width, channels)
                h2, w2, c2 = results[0].orig_img.shape
                
                # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
                mask = cv2.resize(mask_3channel, (w2, h2))

                # Convert BGR to HSV
                # hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

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


                
                indices = np.where(transform != [0])

                # Combine the lists into a list of tuples
                combined_list = list(zip(indices[1],indices[0]))
                combined_list = list(set(combined_list))

                # print("combined_list",combined_list)

                # Initialize empty lists for coordinates on the right side and left side
                right_side = []
                left_side = []
                filtered=[]


                # Generate 25 equally spaced floating-point values between 0 and 800
                floating_values = np.linspace(0, height1, 20)


                # Convert the floating-point values to integers
                y_values = floating_values.astype(int)

                # Iterate through the list of coordinates
                for coord in combined_list:
                    x, y = coord
                    if y in y_values:
                        filtered.append((x,y))
                        sorted_data = sorted(filtered, key=lambda x: x[0])
                        # if x <= v_line_x_pos:
                        #     # Append to the left side list if x <= 100
                        #     left_side.append((x, y))
                        # else:
                        #     # Append to the right side list if x > 100
                        #     right_side.append((x, y))

                # print("filtered ", sorted_data)

                filtered_data = [sorted_data[0]]

                for i in range(1, len(sorted_data)):
                    if sorted_data[i][0] - filtered_data[-1][0] >= 5:
                        filtered_data.append(sorted_data[i])

                # print(filtered_data)   

                # Calculate the differences between consecutive x values
                differences = [filtered_data[i+1][0] - filtered_data[i][0] for i in range(len(filtered_data)-1)]

                # Find the maximum difference
                max_difference_index = differences.index(max(differences))

                # Split the list into two parts
                part1 = filtered_data[:max_difference_index+1]
                part2 = filtered_data[max_difference_index+1:]

                print("Part 1:", part1)
                print("Part 2:", part2)

                left_side=part1
                right_side=part2

                # print("len left", len(left_side), "len right", len(right_side))
                # Initialize an empty list for midpoints
                        
                # right_side=quard_regresion(right_side)
                # # print(right_side)
                # left_side=quard_regresion(left_side)

                midpoints = []

                # Iterate through all pairs of coordinates in left_side and right_side
                for left_coord in left_side:
                    for right_coord in right_side:
                        # Check if the y-coordinates match
                        if left_coord[1] == right_coord[1]:
                            # Calculate the average x-value for the same y-coordinate
                            avg_x = int((left_coord[0] + right_coord[0]) / 2)
                            # Append the midpoint coordinates to the midpoints list
                            midpoints.append((avg_x, left_coord[1]))  # or right_coord[1], as they are the same

                # midpoints=quard_regresion(midpoints)
                # # Print the list of midpoints
                print("Midpoints:", midpoints)
                for a in midpoints:
                    x, y = a
                    if y!=465:
                    # OpenCV uses BGR color format, so (0, 0, 255) represents red color
                        cv2.circle(transform, (x, y), 5, (255, 255, 255), -1)  # -1 specifies to fill the circle
                        cv2.circle(annotated_frame, (x, y), 5, (255, 255, 255), -1)
                
                 # v_line_x_pos=int(width/4)
                v_line_x_pos=int(width1/2)

                cv2.line(transform, (v_line_x_pos, 0), (v_line_x_pos, height), (255,255,255), 2)

                displacement_dict = {y: x - v_line_x_pos for x, y in midpoints}
                displacement_dict = dict(sorted(displacement_dict.items(), key=lambda item: item[0], reverse=True))
                displacement_list = list(displacement_dict.values())
                # print("list", displacement_list)
                
                n= len(displacement_list)
                image_lane=displacement_list

                

                
                # # Define the radius of the dots
                # radius = 5

                # # Draw a dot at each coordinate
                # for coord in combined_list:
                #     x, y = coord
                #     # OpenCV uses BGR color format, so (0, 0, 255) represents red color
                #     cv2.circle(edge, (x, y), radius, (255, 255, 255), -1)  # -1 specifies to fill the circle


                # Show the masked part of the image
                # cv2.imshow("mask", stacked_image)
                cv2.imshow("transpose",transform)

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
            if cv2.waitKey(1) & 0xFF == ord("q"):
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



