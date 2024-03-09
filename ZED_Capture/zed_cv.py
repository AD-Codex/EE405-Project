
import cv2
import numpy

cap = cv2.VideoCapture(0)

if cap.isOpened() == 0:
    exit(-1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while True :
    retval, frame = cap.read()

    left_right_image = numpy.split(frame, 2, axis=1)

    # cv2.imshow("frame", frame)
    # cv2.imshow("right", left_right_image[0])
    cv2.imshow("left", left_right_image[1])
    
    if cv2.waitKey(30) >= 0:
        break


exit(0)


