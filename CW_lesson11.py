import pandas as pd # для роботи з csv файлами
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv('data/figures.csv')
print(df.head())

encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])
X = df[['area', 'perimeter', 'corners']]
y = df['label_enc']

model = keras.Sequential([layers.Dense(8, activation='relu', input_shape=(3, )),
                          layers.Dense(8, activation='relu'),
                          layers.Dense(3, activation='softmax')
                          ])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X, y, epochs=200, verbose=0)

test = np.array([[42, 26, 4]])

pred = model.predict(test)
print(f'Ймовірність кожного класу: {pred}')
print(f'Результат: {encoder.inverse_transform([np.argmax(pred)])}')

plt.plot(history.history['loss'], label='Втрати')
plt.plot(history.history['accuracy'], label='Точність')
plt.xlabel('Епоха')
plt.ylabel('Значення')
plt.title('Навчання моделі')
plt.legend()
plt.show()




