import cv2
import numpy as np


def region_selection(image):
	# create an array of the same size as of the input image 
	mask = np.zeros_like(image) 
	# if you pass an image with more then one channel
	if len(image.shape) > 2:
		channel_count = image.shape[2]
		ignore_mask_color = (255,) * channel_count
	# our image only has one channel so it will go under "else"
	else:
		# color of the mask polygon (white)
		ignore_mask_color = 255
	# creating a polygon to focus only on the road in the picture
	# we have created this polygon in accordance to how the camera was placed
	rows, cols = image.shape[:2]
	bottom_left = [cols * 0.01, rows * 0.95]
	top_left	 = [cols * 0.25, rows * 0.6]
	bottom_right = [cols * 0.9, rows * 0.95]
	top_right = [cols * 0.5, rows * 0.6]
	vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
	# filling the polygon with white color and generating the final mask
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	# performing Bitwise AND on the input image and mask to get only the edges on the road
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image



cap = cv2.VideoCapture("2.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



frame_counter = 0

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame_counter += 1
        # Skip frames if not the 9th frame
        if frame_counter < 6:
            continue

        # Reset the frame counter
        frame_counter = 0
        
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        kernel_size = 5
        # Applying gaussian blur to remove noise from the frames
        blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
       
        low_t = 50
        # second threshold for the hysteresis procedure 
        high_t = 150
        # applying canny edge detection and save edges in a variable
        edges = cv2.Canny(blur, low_t, high_t)

        region=region_selection(edges)

        rho = 1             
        # Angle resolution of the accumulator in radians.
        theta = np.pi/180   
        # Only lines that are greater than threshold will be returned.
        threshold = 200      
        # Line segments shorter than that are rejected.
        minLineLength = 200  
        # Maximum allowed gap between points on the same line to link them
        maxLineGap = 400
        
        cv2.HoughLinesP(region, rho = rho, theta = theta, threshold = threshold, minLineLength = minLineLength, maxLineGap = maxLineGap)
        
        cv2.imshow("frame",frame)
        cv2.imshow("mask",region)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

  
# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()



