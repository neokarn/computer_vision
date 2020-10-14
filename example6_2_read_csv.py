from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import numpy as np

#Create model
input = Input(shape=(5,)) #5 Features
hidden1 = Dense(10, activation='tanh')(input)
hidden2 = Dense(10, activation='tanh')(hidden1)
output = Dense(3, activation='softmax')(hidden2) #Classification (3 classes)
model = Model(inputs=input, outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()

#Read data from file (download at https://github.com/neokarn/computer_vision/blob/master/data.csv)
data = np.asarray([[float(num) for num in line.split(',')] for line in open('data.csv')])
print('---------------- DATA -----------------------------------------')
print(data)

#Train Model
x_train = data[0:120,0:5] # First 80 samples for training
y_train = data[0:120,5]

print('---------------- x_train -----------------------------------------')
print(x_train)
print('---------- y_train before to_categorical()----------------------------------')
print(y_train)

y_train = to_categorical(y_train)
print('---------- y_train after to_categorical()----------------------------------')
print(y_train) #one hot vectors

model.fit(x_train, y_train, epochs=100, batch_size=5, verbose = 0)


#Test Model
x_test = data[120:,0:5]
y_test = data[120:,5]

y_pred = model.predict(x_test)
print('---------- y_pred before argmax()----------------------------------')
print(y_pred)

y_pred = np.argmax(y_pred,axis = -1)
print('---------- y_pred after argmax()----------------------------------')
print(y_pred)

print('---------- Confusion Matrix ----------------------------------')
cm = confusion_matrix(y_test, y_pred) #y_test y_pred class index
print(cm)

print('---------- model.evaluate() ----------------------------------')
y_test = to_categorical(y_test)
score = model.evaluate(x_test, y_test) #y_test one hot
print(score)
