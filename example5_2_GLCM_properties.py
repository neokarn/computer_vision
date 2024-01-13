import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

#download images form https://drive.google.com/file/d/1JfJYr-qJvgt1Jyz-Gop-oRci-TipuQmb/view?usp=sharing

im = cv2.imread("TextureClassification//Beef//1.jpg")
im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cv2.imshow("image",im)

glcm = graycomatrix(im_gray, [5], [0], 256, symmetric=True, normed=True)
#glcm = graycomatrix(im_gray, [5,10,15], [0,np.pi/2], 256, symmetric=True, normed=True)
print('GLCM Shape')
print(glcm.shape)

print('ASM')
print(graycoprops(glcm, 'ASM'))

print('contrast')
print(graycoprops(glcm, 'contrast'))

print('homogeneity')
print(graycoprops(glcm, 'homogeneity'))

print('correlation')
print(graycoprops(glcm, 'correlation'))

cv2.waitKey(0)
cv2.destroyAllWindows()

