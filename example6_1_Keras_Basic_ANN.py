#For Google Colab Version
#https://colab.research.google.com/drive/1NNMXzedIkMiDH9GkiK6Tzp9HB9mjb0bl?usp=share_link

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
import numpy as np

#Create model
input = Input(shape=(3,))
hidden = Dense(4, activation='tanh')(input)
output = Dense(1, activation='sigmoid')(hidden) #Binary Classification
model = Model(inputs=input, outputs=output)

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

#Train model
x_train = np.asarray([[1,0,1],
                      [4, 2, 0],
                      [-4, 0, 1],
                      [1,2,3],
                      [-1,-2,3],
                      [0,-1,3],
                      [1, 0, 0],
                      [-1, 0, -2],
                      [4, -2, 7],
                      [-1,-1,4]])

y_train = np.asarray([1,1,0,1,0,1,0,0,1,0])

model.fit(x_train, y_train, epochs=100, batch_size=10)

#Test model
x_test = np.asarray([[1,5,1],
                     [5,-1,3],
                     [-5,0,1],
                     [0,0,0]])

y_test = np.asarray([1,1,0,0])

y_pred = model.predict(x_test)

print("y_pred")
print(y_pred)

score = model.evaluate(x_test, y_test)

print("score")
print(score)

