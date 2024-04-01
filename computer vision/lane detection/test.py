import cv2
import numpy as np

def draw_parallel_lines(img, point1, point2, distance):
    # Calculate the angle of the line
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    
    # Handle the case of a vertical line
    if dx == 0:
        # Draw parallel vertical lines
        cv2.line(img, (point1[0] - distance, point1[1]), (point1[0] - distance, point2[1]), (255, 0, 0), 2)
        cv2.line(img, (point1[0] + distance, point1[1]), (point1[0] + distance, point2[1]), (255, 0, 0), 2)
    else:
        # Calculate slope and intercept of the line
        slope = dy / dx
        intercept = point1[1] - slope * point1[0]
        
        # Calculate perpendicular slope
        perp_slope = -1 / slope
        
        # Calculate new endpoints for parallel lines
        delta_x = distance / np.sqrt(1 + perp_slope**2)
        delta_y = delta_x * perp_slope
        
        # Calculate coordinates for the two points for each side
        p1_side1 = (int(point1[0] - delta_x), int(point1[1] - delta_y))
        p2_side1 = (int(point2[0] - delta_x), int(point2[1] - delta_y))
        p1_side2 = (int(point1[0] + delta_x), int(point1[1] + delta_y))
        p2_side2 = (int(point2[0] + delta_x), int(point2[1] + delta_y))
        
        # Draw the parallel lines
        cv2.line(img, p1_side1, p2_side1, (255, 0, 0), 2)
        cv2.line(img, p1_side2, p2_side2, (255, 0, 0), 2)

# Create a black image
image = np.zeros((500, 500, 3), dtype=np.uint8)

# Define points and distance
point1 = (100, 100)
point2 = (100, 150)
distance = 50

# Draw main line
cv2.line(image, point1, point2, (0, 255, 0), 2)

# Draw parallel lines
draw_parallel_lines(image, point1, point2, distance)

# Display the image
cv2.imshow("Parallel Lines", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
