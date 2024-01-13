import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

#download images form https://drive.google.com/file/d/1JfJYr-qJvgt1Jyz-Gop-oRci-TipuQmb/view?usp=sharing

im = cv2.imread("TextureClassification//Beef//1.jpg")
im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cv2.imshow("image",im)

L = 10
im_gray = (im_gray/16).astype(np.uint8)
glcm = graycomatrix(im_gray, range(1,L+1), [0,np.pi/4,np.pi/2], 16, symmetric=True, normed=True)
print('Shape of GLCM')
print(glcm.shape)

glcm_props = np.zeros(4*L*3)
glcm_props[0:(L*3)] = graycoprops(glcm, 'ASM').reshape(1,-1)[0]
glcm_props[(L*3):(L*3*2)] = graycoprops(glcm, 'contrast').reshape(1,-1)[0]
glcm_props[(L*3*2):(L*3*3)] = graycoprops(glcm, 'homogeneity').reshape(1,-1)[0]
glcm_props[(L*3*3):(L*3*4)] = graycoprops(glcm, 'correlation').reshape(1,-1)[0]

print('Shape of GLCM Stat Feature')
print(glcm_props.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()
