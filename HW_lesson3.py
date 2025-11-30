import cv2

img = cv2.imread('images/me2.jpg')
img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
cv2.rectangle(img,
              (250, 30),
              (365, 195),
              (0, 0, 255), 1)
cv2.putText(img,
            "Майстренко Роман",
            (250, 225),
            cv2.FONT_HERSHEY_COMPLEX,
            0.75, (0, 0, 255))
cv2.imshow("Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

