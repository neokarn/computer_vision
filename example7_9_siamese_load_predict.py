#Colab Version: https://colab.research.google.com/drive/1iVmXiraU5ei6I4eMDa-KOqkm3vTd5jOX?usp=sharing

from tensorflow.keras.models import load_model
import cv2
import numpy as np

IMAGE_SIZE = (128,128)

model = load_model('image_pair_classify.keras')

# download dataset form https://drive.google.com/drive/folders/1jiwbiSbEMQkVGg3Oq2TZI2M9CPm0cJwC?usp=drive_link
test_im_1 = cv2.imread('/animalfaces/test/RabbitHead/rabbitfrontalfrontal0005.jpg')
true_size = test_im_1.shape
imshow_size = (256,round(true_size[0]*256/true_size[1]))
cv2_imshow(cv2.resize(test_im_1, imshow_size))
test_im_1 = cv2.cvtColor(test_im_1, cv2.COLOR_BGR2RGB)
test_im_1 = cv2.resize(test_im_1, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
test_im_1 = test_im_1/255.
test_im_1 = np.expand_dims(test_im_1, axis=0)

test_im_2 = cv2.imread('/animalfaces/test/RabbitHead/rabbitfrontalfrontal0015.jpg')
true_size = test_im_2.shape
imshow_size = (256,round(true_size[0]*256/true_size[1]))
cv2_imshow(cv2.resize(test_im_2, imshow_size))
test_im_2 = cv2.cvtColor(test_im_2, cv2.COLOR_BGR2RGB)
test_im_2 = cv2.resize(test_im_2, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
test_im_2 = test_im_2/255.
test_im_2 = np.expand_dims(test_im_2, axis=0)

y_pred = model.predict([test_im_1,test_im_2])
print(y_pred)
