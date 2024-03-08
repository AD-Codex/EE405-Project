
import pyzed.sl as sl
import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n.pt')

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720 # Use HD720 opr HD1200 video mode, depending on camera type.
    init_params.camera_fps = 100  # Set fps at 30

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(err)+". Exit program.")
        exit()


    i = 0
    image = sl.Mat()
    depth_zed =sl.Mat(1280, 720, sl.MAT_TYPE.F32_C1)
    depth_image_zed = sl.Mat(1280, 720, sl.MAT_TYPE.U8_C4)

    runtime_parameters = sl.RuntimeParameters()
    while True:
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:

            # Retrieve RGB data
            zed.retrieve_image(image, sl.VIEW.LEFT)

            # Retrieve the normalized depth image
            zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH)

            # Retrieve depth data (32-bit)
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)

            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get the timestamp at the time the image was captured
            # print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image.get_width(), image.get_height(), timestamp.get_milliseconds()))

            # Load capture data
            depth_ocv = depth_zed.get_data()
            image_ndarray = image.get_data()

            # image - 3d array [ 720, 1280, 4] , but need [ 720, 1280, 3]
            image_ndarray = np.delete(image_ndarray, 3, axis=2)


            # Print the depth value at the center of the image
            # depth_ocv.shape = (720,1280) , (y,x)
            center_x = int(len(depth_ocv[0])/2)
            center_y = int(len(depth_ocv)/2)

            image_ndarray = cv2.line(image_ndarray, (500,300), (800,300), (50,50,255), 1)

            image_ndarray = cv2.putText( image_ndarray, str(depth_ocv[300,500]), (500,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (250,50,55), 1, cv2.LINE_AA)
            image_ndarray = cv2.circle( image_ndarray , (500,300), 1, (50,250,55), 2)
            
            image_ndarray = cv2.putText( image_ndarray, str(depth_ocv[300,600]), (600,350), cv2.FONT_HERSHEY_SIMPLEX, 1, (250,50,55), 1, cv2.LINE_AA)
            image_ndarray = cv2.circle( image_ndarray , (600,300), 1, (50,250,55), 2)

            image_ndarray = cv2.putText( image_ndarray, str(depth_ocv[300,700]), (700,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (250,50,55), 1, cv2.LINE_AA)
            image_ndarray = cv2.circle( image_ndarray , (700,300), 1, (50,250,55), 2)

            image_ndarray = cv2.putText( image_ndarray, str(depth_ocv[300,800]), (800,350), cv2.FONT_HERSHEY_SIMPLEX, 1, (250,50,55), 1, cv2.LINE_AA)
            image_ndarray = cv2.circle( image_ndarray , (800,300), 1, (50,250,55), 2)


        current_fps = zed.get_current_fps()
        # print("current fps: ", current_fps)
        image_ndarray = cv2.putText( image_ndarray, str(current_fps), (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,50,255), 1, cv2.LINE_AA)

        cv2.imshow("new", image_ndarray)

        # cv2.imshow( "depth", depth_image_zed.get_data())

        if cv2.waitKey(1) & 0XFF == ord("q"):
            break

    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()