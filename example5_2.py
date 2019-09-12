import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops


im = cv2.imread("C://Users//surface//Google Drive//gmkarn@gmail.com//Computer Vision//2559-01//MATLAB//Texture//Beef//1.jpg")
im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cv2.imshow("image",im)

glcm = greycomatrix(im_gray, [5], [0], 256, symmetric=True, normed=True)
glcm_props = np.zeros(4)
glcm_props[0] = greycoprops(glcm, 'ASM')
glcm_props[1] = greycoprops(glcm, 'contrast')
glcm_props[2] = greycoprops(glcm, 'homogeneity')
glcm_props[3] = greycoprops(glcm, 'correlation')

print(glcm.shape)
print(glcm_props)

cv2.waitKey(0)
cv2.destroyAllWindows()
