
import pyzed.sl as sl
import cv2
from ultralytics import YOLO
import numpy as np
import csv

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
            center_x = int(len(depth_ocv[0])/2)
            center_y = int(len(depth_ocv)/2)

            #YOLO model load
            results = model.predict(image_ndarray, conf=0.5)
            frame_edit = image_ndarray
            for r in results:
                # print("box :", r.boxes.xyxy)
                boxes = r.boxes.cuda()
                object_XY = boxes.xyxy
                for i in range(len(object_XY)):
                    start_point_x = int(object_XY[i][0])
                    start_point_y = int(object_XY[i][1])
                    end_point_x = int(object_XY[i][2])
                    end_point_y = int(object_XY[i][3])

                    start_point = ( start_point_x, start_point_y)
                    end_point = ( end_point_x, end_point_y)

                    print( r.names[int(r.boxes.cls[i])], ":", start_point, end_point)
                    # depth_ocv.shape - (720, 1280) ; (y,x)
                    depth_read_data = depth_ocv[start_point_y:end_point_y, start_point_x:end_point_x]
                    # print(depth_ocv[start_point_y:end_point_y, start_point_x:end_point_x])
                    
                    frame_edit = cv2.rectangle(frame_edit, start_point, end_point, (255,0,0), 1)
                    frame_edit = cv2.putText( frame_edit, str(r.names[int(r.boxes.cls[i])]), start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (50,50,255), 1, cv2.LINE_AA)


            # print(depth_ocv[int(len(depth_ocv)/2)][int(len(depth_ocv[0])/2)])

        
        current_fps = zed.get_current_fps()
        # print("current fps: ", current_fps)
        image_ndarray = cv2.putText( image_ndarray, str(current_fps), (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,50,255), 1, cv2.LINE_AA)

        # cv2.imshow("new", image_ndarray)
        cv2.imshow("YOLO", frame_edit)

        # cv2.imshow( "depth", depth_image_zed.get_data())

        if cv2.waitKey(1) & 0XFF == ord("q"):
            file = open('csv_fle.csv', 'w')
            writer = csv.writer(file)
            writer.writerows(depth_read_data)
            break

    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()