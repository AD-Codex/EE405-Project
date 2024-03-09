import cv2
from ultralytics import YOLO
import numpy as np
from linear_regression import *

# points = [(1, 2), (4, 2), (4, 3)]
# print(quard_regresion(points))


model = YOLO('yolov8n-seg.pt')
model1=YOLO('best.pt')

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(2)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# v_line_x_pos=int(width/4)
v_line_x_pos=int(width/2)

# Generate 25 equally spaced floating-point values between 0 and 800
floating_values = np.linspace(0, height, 25)

# Convert the floating-point values to integers
y_values = floating_values.astype(int)

frame_counter = 0
gradients = []

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

        

        
            
        results = model1.predict(frame, conf=0.1,max_det=1)
###################
        # # Define the vertices of the trapezium
        # vertices = np.array([[10, height], [int(width/2)-500, height], [int(width/2)+500, height], [250, 200]], dtype=np.int32)

        # # Reshape the vertices array to fit OpenCV's requirements
        # vertices = vertices.reshape((-1, 1, 2))

        # # Draw the trapezium on the image
        # cv2.polylines(frame, [vertices], isClosed=True, color=(0, 255, 0), thickness=2)
####################

        try:
            # Run YOLOv8 inference on the frame
            annotated_frame = results[0].plot(boxes=False)

            cv2.line(annotated_frame, (v_line_x_pos, 0), (v_line_x_pos, height), (255,0,0), 2)

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
                hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

                # Define range of brightness in HSV
                lower_black = np.array([0,0,0])
                upper_black = np.array([0,0,1])

                # Create a mask. Threshold the HSV image to get everything black
                mask = cv2.inRange(mask, lower_black, upper_black)

                # Invert the mask to get everything but black
                mask = cv2.bitwise_not(mask)

                cv2.imshow("mask", mask)
                # cv2.imwrite('mask.jpg', mask)
                
                # Applying the Canny Edge filter
                edge = cv2.Canny(mask, 50, 150)

                indices = np.where(edge != [0])

                # Combine the lists into a list of tuples
                combined_list = list(zip(indices[1],indices[0]))
                combined_list = list(set(combined_list))

                #####################################
                # print("points")
                mid_road1=[]
                for xi in range(width):
                    y_val=[]
                    for xn,yn in combined_list:
                        if xn==xi:
                            y_val.append(yn)
                    
                    if len(y_val)>0:
                        average1 = sum(y_val) / len(y_val)
                        # print(average1)

                        mid_road1.append((xi,int(average1)))

                for a in mid_road1:
                    x, y = a
                    cv2.circle(annotated_frame, (x, y), 2, (255, 0, 0), -1)  # -1 specifies to fill the circle
                

                mid_road2=[]
                for yi in range(height):
                    x_val=[]
                    for xn,yn in combined_list:
                        if yn==yi:
                            x_val.append(xn)
                    
                    if len(x_val)>0:
                        average2 = sum(x_val) / len(x_val)
                        # print(average1)

                        mid_road2.append((int(average2),yi))

                for a in mid_road2:
                    x, y = a
                    cv2.circle(annotated_frame, (x, y), 2, (0, 150, 255), -1)  # -1 specifies to fill the circle
                
                
                ################################################

                # Initialize empty lists for coordinates on the right side and left side
                right_side = []
                left_side = []

                # Iterate through the list of coordinates
                for coord in combined_list:
                    x, y = coord
                    if y in y_values:
                        if x <= v_line_x_pos:
                            # Append to the left side list if x <= 100
                            left_side.append((x, y))
                        else:
                            # Append to the right side list if x > 100
                            right_side.append((x, y))

                # print("len left", len(left_side), "len right", len(right_side))
                # Initialize an empty list for midpoints
                        
                # right_side=quard_regresion(right_side)
                # # print(right_side)
                # left_side=quard_regresion(left_side)

                midpoints1 = []

                # Iterate through all pairs of coordinates in left_side and right_side
                for left_coord in left_side:
                    for right_coord in right_side:
                        # Check if the y-coordinates match
                        if left_coord[1] == right_coord[1]:
                            # Calculate the average x-value for the same y-coordinate
                            avg_x = int((left_coord[0] + right_coord[0]) / 2)
                            # Append the midpoint coordinates to the midpoints list
                            midpoints1.append((avg_x, left_coord[1]))  # or right_coord[1], as they are the same

                midpoints1=quard_regresion(midpoints1)

                for a in midpoints1:
                    x, y = a
                    cv2.circle(annotated_frame, (x, y), 2, (255, 255, 255), -1)  # -1 specifies to fill the circle
                

                # print("com ", combined_list)


                # # Calculate the change in y and x
                # delta_y = midpoints1[-1][1] - midpoints1[0][1]
                # delta_x = midpoints1[-1][0] - midpoints1[0][0]

                # # Calculate the gradient (change in y / change in x)
                # gradient = delta_y / delta_x


                # # Append the gradient to the list
                # gradients.append(gradient)
                # if len(gradients)>3:
                #     gradients.pop(0)
                #     if (gradients[1]-gradients[0])+0.3<(gradients[2]-gradients[1]):
                #         # devide points
                #         # print("change detected")

                #         ############################################################################
                #         # Define the text, font, and position
                #         # text = "change detected"
                #         # font = cv2.FONT_HERSHEY_SIMPLEX
                #         # position = (50, 50)  # Coordinates (x, y) from the top-left corner
                #         # font_scale = 1
                #         # font_color = (255, 255, 255)  # BGR color format
                #         # line_thickness = 2

                #         # # Draw the text on the frame
                #         # cv2.putText(annotated_frame, text, position, font, font_scale, font_color, line_thickness)

                #         # cv2.putText(annotated_frame, str(midpoints[0]), midpoints[0], font, font_scale, font_color, line_thickness)

                #         ############################################################################

                        
                        


                        
                # print("gra "+ str(gradients))

                displacement_dict = {y: x - v_line_x_pos for x, y in midpoints1}
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
                # cv2.imshow("mask", edge)

            cv2.imshow("YOLOv8 Inference2",annotated_frame)

            # object detection
            results0 = model.predict(annotated_frame, conf=0.25, classes=[0,1,2,3,5,7,9,11,15,16])
            try:
                annotated_frame1 = results0[0].plot(boxes=False)
            
                cv2.imshow("YOLOv8 Inference2",annotated_frame1)
            except:
                print('no object')
                cv2.imshow("YOLOv8 Inference2",annotated_frame)

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



