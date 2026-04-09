import cv2
from ultralytics import YOLO
import numpy as np
import time
import webcolors

cars_detected = 0

def detect_cars(image):
    global cars_detected
    track_id = 0
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    for detection in results[0].boxes:
        class_id = int(detection.cls)
        label = model.names[class_id]
        if label.lower() == "car" or label.lower() == "bus" or label.lower() == "truck":
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                mean_b, mean_g, mean_r = cv2.mean(roi)[:3]
                mean_color = (int(mean_r), int(mean_g), int(mean_b))
                color_name = get_color_name(mean_color)
                color_text = color_name

                swatch_top_left = (x1, y2 + 5)
                swatch_bottom_right = (x1 + 30, y2 + 35)
                cv2.rectangle(frame, swatch_top_left, swatch_bottom_right, (int(mean_b), int(mean_g), int(mean_r)), -1)
                cv2.rectangle(frame, swatch_top_left, swatch_bottom_right, (255, 255, 255), 1)
                cv2.putText(frame, color_text, (x1 + 40, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            #cv2.putText(frame, f'Color:{detect_car_color(cropped_car)}', (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            if y2 > line_y and track_id not in crossed_ids:
                crossed_ids.append(track_id)
                cars_detected += 1
            track_id += 1






def closest_color(requested_color):
    min_colors = {}
    for name in webcolors.names("css3"):
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_color_name(requested_color):
    try:
        # Try exact match first
        return webcolors.rgb_to_name(requested_color)
    except ValueError:
        # Return closest if exact match fails
        return closest_color(requested_color)




crossed_ids = []
line_y = 0
cars_detected = 0

VIDEO_PATH = "video/cars.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
model = YOLO('yolov8n.pt')
CONF_THRESHOLD = 0.35
RESIZE_WIDTH = 960 #None


prev_time = time.time()
fps = 0.0

pseudo_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        new_w = int(scale * w)
        new_h = int(scale * h)
    frame = cv2.resize(frame, (new_w, new_h))
    line_y = frame.shape[0] - 100
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)

    detect_cars(frame)
    cv2.putText(frame, f'Cars:{cars_detected}', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    cv2.imshow('YOLO', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()