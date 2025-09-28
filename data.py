import cv2
from ultralytics import YOLO

model = YOLO('yolov8m.pt')  
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    found_labels = []

    for box in results.boxes:
        if box.conf < 0.5:
            continue

        cls_id = int(box.cls)
        label = model.names[cls_id]
        if label.lower() in ['cell phone', 'mobile phone', 'smartphone']:
            found_labels.append(label)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label_en = 'Cell Phone'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label_en, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if found_labels:
        text = f"Detected: {len(found_labels)} cell phone(s)"
    else:
        text = "No cell phone detected"

    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    cv2.imshow('Cell Phone Detection via Webcam', frame)

    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()