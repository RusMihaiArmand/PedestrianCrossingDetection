from ultralytics import YOLO
import cv2



confidence_min = 0.6

model = YOLO("my_model.pt")
target_classes = ['pedestrian_light', 'crossing'] 

image_path = "image1.png"
results = model(image_path)
image = cv2.imread(image_path)


for res in results:
    for box in res.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        confidence = box.conf[0]

        if label in target_classes and confidence >= confidence_min:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} - {100*confidence:.2f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            print(f"Detected {label}. Certitude: {100*confidence:.2f}%")


cv2.imwrite("image1_output.png", image)

