import cv2

# img = cv2.imread('images/1.png')
# img = cv2.resize(img, (500, 300))
# cv2.imshow('image', img)
cap = cv2.VideoCapture('video/video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (600, 300))
    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()