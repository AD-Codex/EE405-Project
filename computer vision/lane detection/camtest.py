import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model1=YOLO('best.pt')

cap = cv2.VideoCapture(2)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



frame_counter = 0

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    # height,width=frame.shape[:2]
    frame=frame[:height,:int(width/2)]


    if success:
        frame_counter += 1
        # Skip frames if not the 9th frame
        if frame_counter < 6:
            continue

        # Reset the frame counter
        frame_counter = 0


        
            
        results = model1.predict(frame, conf=0.1,max_det=1)
        
        try:
            # Run YOLOv8 inference on the frame
            
            annotated_frame = results[0].plot()

            mask=results[0].masks.data.cpu().numpy()[0]
            # cv2.imshow("YOLOv8 Inference2",annotated_frame)

            # object detection
            results0 = model.predict(annotated_frame, conf=0.25)
            try:
                annotated_frame1 = results0[0].plot()
            
                cv2.imshow("YOLOv8 Inference2",annotated_frame1)
            except:
                print('no object')
                cv2.imshow("YOLOv8 Inference2",annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except:
            # cv2.imshow("YOLOv8 Inference2",frame)
            print("except part is running")
            cv2.imshow("YOLOv8 Inference2",frame)



        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break



        

        
        
# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()



