#For Goole Colab Version
#https://colab.research.google.com/drive/1UwO27IYQVmsa-DD4sbxN7FdK_yfFAVkV?usp=share_link

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import cv2

#Create model
input = Input(shape = (50,50,1))
conv1 = Conv2D(10,3,activation='relu')(input)
pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(20,3,activation='relu')(pool1)
pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
flat = Flatten()(pool2)
hidden = Dense(12, activation='relu')(flat)
output = Dense(3, activation='softmax')(hidden)
model = Model(inputs=input, outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


#Read data from file (download form https://drive.google.com/file/d/1UACFvQ8QCFUQaBWz9umQxbUWLSQlh1co/view?usp=sharing)
N = 30
x_train = np.zeros((N,50,50,1),'float')
y_train = np.zeros((N),'float')
count = 0
for class_id in range(1,4):
    for im_id in range(1,11):
        im = cv2.imread("thainum123/train/"+str(class_id)+"/"+str(im_id)+".bmp",cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im,(50,50))
        x_train[count,:,:,0] = im/255.
        y_train[count] = class_id-1
        count += 1

y_train = to_categorical(y_train)

#Train Model
h = model.fit(x_train, y_train, epochs=20)

plt.plot(h.history['accuracy'])

#Test Model
N = 15
x_test = np.zeros((N,50,50,1),'float')
y_test = np.zeros((N),'float')
count = 0
for class_id in range(1,4):
    for im_id in range(1,6):
        im = cv2.imread("thainum123/test/"+str(class_id)+"/"+str(im_id)+".bmp",cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im,(50,50))
        x_test[count,:,:,0] = im/255.
        y_test[count] = class_id-1
        count += 1

y_test = to_categorical(y_test)

score = model.evaluate(x_test, y_test)
print('score (cross_entropy, accuracy):\n',score)

y_pred = model.predict(x_test)
print('confidence:\n', y_pred)
print('predicted class name:\n', np.argmax(y_pred,axis = -1)+1)

cm = confusion_matrix(np.argmax(y_test,axis = -1), np.argmax(y_pred,axis = -1))
print("Confusion Matrix:\n",cm)

plt.show()
