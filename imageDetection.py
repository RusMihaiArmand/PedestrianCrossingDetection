import json
from ultralytics import YOLO
import cv2




confidence_min = 0.6
image_path = "image.png"




with open("objectsData.json", 'r') as file:
    objectsData = json.load(file)

with open("messageData.json", 'r') as file:
    messageData = json.load(file)

model = YOLO("my_model_v3.pt")
target_classes = ['pedestrian_light', 'crossing', 'pedestrian_light_green', 'pedestrian_light_red'] 


results = model(image_path)
image = cv2.imread(image_path)

detected_labels = []

for res in results:
    for box in res.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        confidence = box.conf[0]

        if label in target_classes and confidence >= objectsData.get(label,{}).get('minimum_confidence', confidence_min):
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if(objectsData.get(label,{}).get('display', True)):

                line_type_obj = objectsData.get(label,{}).get('line_type', {})
                line_type = None

                if line_type_obj and hasattr(cv2, line_type_obj):
                    line_type = getattr(cv2, line_type_obj)

                cv2.rectangle(image, (x1, y1), (x2, y2), (objectsData.get(label,{}).get('B', 255),objectsData.get(label,{}).get('G', 255),objectsData.get(label,{}).get('R', 255))
                            , objectsData.get(label,{}).get('line_thickness', 1), lineType= line_type)
                cv2.putText(image, f"{objectsData.get(label,{}).get('display_name', label)} - {100*confidence:.2f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                            (objectsData.get(label,{}).get('B', 255),objectsData.get(label,{}).get('G', 255),objectsData.get(label,{}).get('R', 255)), 2)
            print(f"Detected {objectsData.get(label,{}).get('display_name', label)}. Certitude: {100*confidence:.2f}%")

        detected_labels.append(label)


    


msg = "-"

if 'crossing' in detected_labels:
    if 'pedestrian_light' in detected_labels:
        if 'pedestrian_light_green' in detected_labels:
            msg = "light_crossing_green"
        elif 'pedestrian_light_red' in detected_labels:
            msg = "light_crossing_red"
        else:
            msg = "light_crossing"
    else:
        msg = "simple_crossing"
        


if messageData.get(msg, {}):
    font_type = cv2.FONT_HERSHEY_SIMPLEX
    font_type_obj = messageData.get(msg,{}).get('font', {})

    if font_type_obj and hasattr(cv2, font_type_obj):
        font_type = getattr(cv2, font_type_obj)


    (text_width, text_height), _ = cv2.getTextSize(messageData.get(msg, {}).get('message', ' '), font_type, messageData.get(msg, {}).get('scale', 2), messageData.get(msg, {}).get('thickness', 1))
    

    cv2.rectangle(image, (5, 5), (text_width + 10, text_height + 10), 
                  (messageData.get(msg,{}).get('background_color',{}).get('B', 255),messageData.get(msg,{}).get('background_color',{}).get('G', 255),messageData.get(msg,{}).get('background_color',{}).get('R', 255)), thickness=-1)
    cv2.putText(image, messageData.get(msg, {}).get('message', ' '), (10, 10+text_height), font_type, messageData.get(msg, {}).get('scale', 2), 
                (messageData.get(msg,{}).get('text_color',{}).get('B', 0),messageData.get(msg,{}).get('text_color',{}).get('G', 0),messageData.get(msg,{}).get('text_color',{}).get('R', 0)), messageData.get(msg, {}).get('thickness', 1))



cv2.imwrite("image_output.png", image)

