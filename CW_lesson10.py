import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def generate_image(color, shape):
    img = np.zeros((200, 200, 3), np.uint8)
    if shape == "circle":
        cv2.circle(img, (100, 100), 50, color, -1)
    elif shape == "square":
        cv2.rectangle(img, (50, 50), (150, 150), color, -1)
    elif shape == "triangle":
        points = np.array([[100, 40], [40, 160], [160, 100]])
        cv2.drawContours(img, [points], 0, color, -1)
    return img

FEATURES = []
labels = []

colors = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
}

shapes = ['square', 'circle', 'triangle']

for color_name, bgr in colors.items():
    for shape in shapes:
        for _ in range(10):
            img = generate_image(bgr, shape)
            mean_color = cv2.mean(img)[:3] # mean видає значення BGRA - Alpha не потрібно
            predict = [mean_color[0], mean_color[1], mean_color[2]]
            FEATURES.append(predict)
            labels.append(f'{color_name}_{shape}')



FEATURES_train, FEATURES_test, labels_train, labels_test = train_test_split(FEATURES, labels, test_size=0.3, stratify=labels)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(FEATURES_train, labels_train)
accuracy = model.score(FEATURES_test, labels_test)
print(f'Accuracy: {round(accuracy*100, 2)}%')

test_img = generate_image((0, 0, 255), 'square')
mean_color = cv2.mean(test_img)[:3]
predict = model.predict([mean_color])
print(f'Prediction: {predict}')
cv2.imshow('image', test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
