import cv2
import numpy as np


# Load an image from file
image = cv2.imread('mask.jpg',0)

# Check if the image was successfully loaded
if image is not None:
    # Display the image in a window
    cv2.imshow('Image', image)
    edges= left_edge(image)
    cv2.imshow('left', left)

    # Wait for any key to be pressed
    cv2.waitKey(0)
    # Close all OpenCV windows
    cv2.destroyAllWindows()
else:
    print("Error: Unable to load image.")
