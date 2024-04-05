import cv2
import numpy as np


frame= cv2.imread("1.png")

def perspective(frame):

    # frame= cv2.resize(frame,(width,height))
    try:
        height, width,channel = frame.shape
        black_image = np.zeros((height, width,channel), dtype=np.uint8)
    except:
        height, width = frame.shape
        black_image = np.zeros((height, width), dtype=np.uint8)

    # Create a black image
    

    # Horizontally stack the black image to the frame
    stacked_image = np.hstack((frame, black_image))
    stacked_image = np.hstack((black_image, stacked_image))


    tl=[width,0]
    bl=[0,height]
    tr=[2*width,0]
    br=[3*width,height]

    cv2.circle(stacked_image,tl,5,(0,0,255),-1)
    cv2.circle(stacked_image,bl,5,(0,0,255),-1)
    cv2.circle(stacked_image,tr,5,(0,0,255),-1)
    cv2.circle(stacked_image,br,5,(0,0,255),-1)



    pts1=np.float32([tl,bl,tr,br])
    pts2=np.float32([[0,0],[0,height],[width,0],[width,height]])

    matrix= cv2.getPerspectiveTransform(pts1,pts2)
    transform= cv2.warpPerspective(stacked_image,matrix,(width,height))

    return transform


def inv_perspective(frame):

    # frame= cv2.resize(frame,(width,height))
    try:
        height, width,channel = frame.shape
        black_image = np.zeros((height, width,channel), dtype=np.uint8)
    except:
        height, width = frame.shape
        black_image = np.zeros((height, width), dtype=np.uint8)


    # Horizontally stack the black image to the frame
    stacked_image = np.hstack((frame, black_image))
    stacked_image = np.hstack((black_image, stacked_image))


    tl=[0,0]
    bl=[0,height]
    tr=[width,0]
    br=[width,height]

    # cv2.circle(stacked_image,tl,5,(0,0,255),-1)
    # cv2.circle(stacked_image,bl,5,(0,0,255),-1)
    # cv2.circle(stacked_image,tr,5,(0,0,255),-1)
    # cv2.circle(stacked_image,br,5,(0,0,255),-1)



    pts1=np.float32([tl,bl,tr,br])
    pts2=np.float32([[width,0],[0,height],[2*width,0],[3*width,height]])

    matrix= cv2.getPerspectiveTransform(pts1,pts2)
    inv_transform= cv2.warpPerspective(stacked_image,matrix,(width,height))

    return inv_transform


# # cv2.imshow("a",stacked_image)
cv2.imshow("trans",perspective(frame))


cv2.imshow("inverse",inv_perspective(perspective(frame)))

cv2.waitKey(0)