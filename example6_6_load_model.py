from keras.models import load_model
from sklearn.metrics import confusion_matrix
import numpy as np

#Load model
model = load_model('my_model.h5')

model.summary()

#Read data from file
data = np.asarray([[float(num) for num in line.split(',')] for line in open('data.csv')])

#Test Model
x_test = data[120:,0:5]
y_test = data[120:,5]

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis = -1)
cm = confusion_matrix(y_test, y_pred)
print(cm)
