from keras.models import Sequential
from keras.layers import Input, Dense
from keras.utils import to_categorical
import numpy as np

#Create model by using sequential structure
model = Sequential()
model.add(Dense(5, input_dim=5, activation='tanh'))
model.add(Dense(5, activation='tanh'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#Read data from file
data = np.asarray([[float(num) for num in line.split(',')] for line in open('data.csv')])

#Train Model
x_train = data[0:80,0:5]
y_train = data[0:80,5]
y_train = to_categorical(y_train)

model.fit(x_train, y_train, epochs=100, batch_size=5)

#save model 
model.save('my_model.h5')
