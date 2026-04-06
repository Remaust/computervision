import os
import cv2
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, 'video')

VIDEO_PATH = os.path.join(VIDEO_DIR, 'video_catdog2.mp4')
cap = cv2.VideoCapture(VIDEO_PATH)

RESIZE_WIDTH = 960

model = YOLO('yolov8n.pt')
CONF_THRESHOLD = 0.4

CAT_CLASS_ID = 15
DOG_CLASS_ID = 16

prev_time = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    scale = RESIZE_WIDTH / w
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    now = time.time()
    dt = now - prev_time
    prev_time = now
    if dt > 0:
        fps = 1.0 / dt

    result = model(frame, conf=CONF_THRESHOLD, verbose=False)
    cat_count = 0
    dog_count = 0

    for r in result:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == CAT_CLASS_ID:
                cat_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'cat: {conf:.2f}', (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)

            elif cls == DOG_CLASS_ID:
                dog_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'dog: {conf:.2f}', (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    total_animals = cat_count + dog_count
    cv2.putText(frame, f'cats: {cat_count}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(frame, f'dogs: {dog_count}', (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(frame, f'total: {total_animals}', (20, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0, 165, 255), 2)

    cv2.imshow("cats and dogs", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
