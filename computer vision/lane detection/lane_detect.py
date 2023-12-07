import cv2
from ultralytics import YOLO



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
cap = cv2.VideoCapture("1.MOV")

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


        for i in range(int(len(results[0].masks.xy[0]))):
            x1, y1 = results[0].masks.xy[0][i]

            if x1<=v_line_x_pos:
                left[floor_to_nearest_5(y1)]=x1

            else:
                right[floor_to_nearest_5(y1)]=x1


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

        print(length_of_displacement_list)
        print("displacement ",dispacement_dic)
        print("displacement list",displacement_list)


        # add any thing here

        # following codes for display in a window
        annotated_frame = cv2.resize(annotated_frame, (640, 480))

        cv2.imshow("YOLOv8 Inference2",annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(0) & 0xFF == ord("q"): #........................change cv2.waitKey(1) to play video auto
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()



