import cv2
import numpy as np

image_path = 'images/fridge.jpg'
img = cv2.imread(image_path)

img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
output_img = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_eq = cv2.equalizeHist(gray)
blurred = cv2.GaussianBlur(gray_eq, (3, 3), 0)

edges = cv2.Canny(blurred, 125, 200)



kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=3)
eroded = cv2.erode(dilated, kernel, iterations=2)

contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

sticker_count = 0
min_area = 2000
max_area = 40000

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h

    if area < min_area or area > max_area:
        continue

    if w > 4 * h or h > 4 * w:
        continue

    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    sticker_count += 1

print(f"Кількість знайдених об'єктів: {sticker_count}")
cv2.imshow("Result", output_img)
cv2.imwrite("images/fridge_output.jpg", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()