import pyzed.sl as sl

image_zed = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)

image_ocv = image_zed.get_data()

