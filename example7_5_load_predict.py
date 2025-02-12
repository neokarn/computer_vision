#For Goole Colab Version
#https://colab.research.google.com/drive/1N-Rt1aVclWyZWcjPG-UZxqmKmOleDTOo?usp=share_link

from tensorflow.keras.models import load_model
import numpy as np
import cv2


BATCH_SIZE = 5
IMAGE_SIZE = (256,256)

#Download dataset form https://drive.google.com/drive/folders/1bbTkgKpQca87S8K_dNoKu2hEPoEqzzDG?usp=sharing

model = load_model('rice_best.keras')

imgfile = 'rice/images/001_t.bmp'
test_im = cv2.imread(imgfile, cv2.IMREAD_COLOR)
test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
test_im = cv2.resize(test_im, IMAGE_SIZE)
test_im = test_im / 255.
test_im = np.expand_dims(test_im, axis=0)
w_pred = model.predict(test_im)
print(imgfile, " = ", w_pred[0][0])
