#For Google Colab Version
#https://colab.research.google.com/drive/1oz8shKHjbXwDdqui_GtvQKvmT3vesTZW?usp=share_link

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

#Create model by using sequential structure
model = Sequential()
model.add(Dense(10, input_dim=5, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#Read data from file (download at https://github.com/neokarn/computer_vision/blob/master/data.csv)
#data = np.asarray([[float(num) for num in line.split(',')] for line in open('https://github.com/neokarn/computer_vision/blob/master/data.csv')])
data = pd.read_csv('https://raw.githubusercontent.com/neokarn/computer_vision/53a504e70033f8addfbf4e019f7d89195ac8a101/data.csv',header=None)
data = np.array(data)


#Train Model
x_train = data[0:120,0:5]
y_train = data[0:120,5]
y_train = to_categorical(y_train)

model.fit(x_train, y_train, epochs=100, batch_size=5)


#Test Model
x_test = data[120:,0:5]
y_test = data[120:,5]

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis = -1)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)
