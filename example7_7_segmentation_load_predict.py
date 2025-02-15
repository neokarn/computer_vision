#Colab Version: https://colab.research.google.com/drive/1994MoQLb_weGwOtCBpbTosSvPUmnM0Tt?usp=sharing

from tensorflow.keras.models import load_model
import cv2
import numpy as np

IMAGE_SIZE = (256,256)

model = load_model('/gdrive/MyDrive/text_localize_model.keras')

# download dataset form https://drive.google.com/drive/folders/1etiED1j8f65YJktni36t7Wdj6M0lRZPy?usp=sharing
test_im = cv2.imread('/textlocalize/validation/Input/001.jpg')
true_size = test_im.shape
imshow_size = (512,round(true_size[0]*512/true_size[1]))
cv2.imshow('Input',cv2.resize(test_im, imshow_size))

test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
test_im = cv2.resize(test_im, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
test_im = test_im/255.
test_im = np.expand_dims(test_im, axis=0)
segmented = model.predict(test_im)
#segmented = np.around(segmented)
segmented = (segmented[0, :, :, 0]*255).astype('uint8')
cv2.imshow('Output',cv2.resize(segmented, imshow_size))
