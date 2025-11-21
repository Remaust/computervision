import cv2, numpy as np
image = cv2.imread("images/1.jpg")

image = cv2.resize(image, (image.shape[1]//3, image.shape[0]//3))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image.shape) # y, x, color channels number
image = cv2.Canny(image, 100, 100)


kernel = np.ones((3, 3), np.uint8)
image = cv2.dilate(image, kernel, iterations=1) #dillation enlarges light zones on an image
image = cv2.erode(image, kernel, iterations=1)
imagenigger = cv2.imwrite("images/2.jpg", image)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()