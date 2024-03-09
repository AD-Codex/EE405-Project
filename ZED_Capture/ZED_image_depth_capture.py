
import pyzed.sl as sl
import cv2
from ultralytics import YOLO

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
            print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image.get_width(), image.get_height(), timestamp.get_milliseconds()))

            # Load depth data into a numpy array
            depth_ocv = depth_zed.get_data()
            # Print the depth value at the center of the image
            print(depth_ocv[int(len(depth_ocv)/2)][int(len(depth_ocv[0])/2)])


            i = i + 1

        current_fps = zed.get_current_fps()
        print("current fps: ", current_fps)

        cv2.imshow( "frame", image.get_data())
        cv2.imshow( "depth", depth_image_zed.get_data())

        if cv2.waitKey(1) & 0XFF == ord("q"):
            break

    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()