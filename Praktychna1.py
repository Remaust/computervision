import cv2
import numpy as np


img = cv2.imread("images/candy.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


lower_orange = np.array([11, 100, 100])
upper_orange = np.array([25, 255, 255])

lower_yellow = np.array([26, 100, 100])
upper_yellow = np.array([35, 255, 255])

lower_violet = np.array([120, 80, 80])
upper_violet = np.array([145, 255, 255])


mask1 = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask2 = cv2.inRange(hsv, lower_orange, upper_orange)
mask3 = cv2.inRange(hsv, lower_violet, upper_violet)

result = mask1 | mask2 | mask3

contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
candy_counter = 0

for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            candy_counter += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            text_y = y - 10 if y - 10 > 20 else y + 10
            text = f"Candy {candy_counter}"
            cv2.putText(img, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)


cv2.imshow("Image", img)
cv2.imshow("Mask", result)
imageresult = cv2.imwrite("images/result.jpg", img)
while 1:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
