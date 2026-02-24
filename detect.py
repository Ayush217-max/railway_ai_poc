from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')  # Load the YOLOv8n model
cap = cv2.VideoCapture(0)  # Open the default camera
while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        break
    results = model(frame)  # Perform object detection on the frame
    annotated = results[0].plot()  # Get the annotated frame with detections
    cv2.imshow('Detection', annotated)  # Display the annotated frame
    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows