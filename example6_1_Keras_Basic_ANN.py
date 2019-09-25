from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import plot_model
import numpy as np

#Create model
input = Input(shape=(3,))
hidden = Dense(4, activation='tanh')(input)
output = Dense(2, activation='softmax')(hidden) #Classification (2 classes)
model = Model(inputs=input, outputs=output)

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#for plot_model
#install Graphviz (https://www.graphviz.org/)
#set PATH to Graphviz\bin (eg. C:\Program Files (x86)\Graphviz2.38\bin)
#install pydot (https://pypi.org/project/pydot/)
#conda install -c anaconda pydot
#pip install pydot

#plot_model(model, to_file='model1.png')



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
y_train = np.asarray([[1,0],
                      [1, 0],
                      [0, 1],
                      [1,0],
                      [0,1],
                      [1,0],
                      [0,1],
                      [0,1],
                      [1,0],
                      [0,1]])

model.fit(x_train, y_train, epochs=100, batch_size=10)

#Test model
x_test = np.asarray([[1,5,1],
                     [5,-1,3],
                     [-5,0,1],
                     [0,0,0]])

y_test = np.asarray([[1,0],
                    [1,0],
                    [0,1],
                    [0,1]])

y_pred = model.predict(x_test)

print(y_pred)

score = model.evaluate(x_test, y_test)

print(score)
