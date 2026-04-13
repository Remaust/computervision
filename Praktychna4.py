import cv2
import os
import yt_dlp
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
YOUTUBE = 'https://www.youtube.com/watch?v=M3EYAY2MftI'
MODEL_PATH = 'yolo26m.pt'
track_history = {}
PPM = 8


def get_stream_url(url):
    ydl_opts = {
        'format': 'bestvideo[height<=480][ext=mp4]/best[height<=480]/worst',
        'quiet': True,
        'no_warnings': True,
    }
    print(f"Connecting to YouTube with yt-dlp...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info['url']
    except Exception as e:
        print(f" Error: {e}")
        return None


model = YOLO(MODEL_PATH)
stream = get_stream_url(YOUTUBE)
if not stream:
    exit()
cap = cv2.VideoCapture(stream)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    result = model.track(frame, classes=[2], conf=0.2, verbose=False)
    if result[0].boxes.id is not None:
        boxes = result[0].boxes.xywh.cpu().numpy()
        track_ids = result[0].boxes.id.int().cpu().tolist()
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            if track_id in track_history:
                prev_x, prev_y = track_history[track_id]
                dist_pix = ((x-prev_x)**2 + (y-prev_y)**2)**0.5
                speed_m = (dist_pix / PPM) * fps
                speed_km = speed_m * 3.6
                cv2.putText(frame, f'id: {track_id}; speed: {int(speed_km)} km/h', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            track_history[track_id] = (x, y)
    car_frame = result[0].plot()
    cv2.imshow('YOLO Cars', car_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
