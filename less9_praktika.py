import cv2
import numpy as np
import os

face_net = cv2.dnn.readNetFromCaffe('dnn/deploy.prototxt','dnn/res10_300x300_ssd_iter_140000.caffemodel')
eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')

input_folder = "images"
output_folder = "output"
formats = ('.jpg', '.jpeg', '.png', '.webp', '.tiff')

os.makedirs(output_folder, exist_ok=True)
files = sorted(os.listdir(input_folder))
# start_index = files.index(0)
for file in files:
    if not file.lower().endswith(formats):
        continue
    path = os.path.join(input_folder, file)
    frame = cv2.imread(path)
    if frame is None:
        continue

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # поріг впевненості
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")

            x, y = max(0, x), max(0, y)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

            roi_gray = cv2.cvtColor(frame[y:y2, x:x2], cv2.COLOR_BGR2GRAY)
            roi_color = frame[y:y2, x:x2]
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(15, 15)
            )
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    output_path = os.path.join(output_folder, file)
    cv2.imwrite(output_path, frame)
