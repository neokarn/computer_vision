import numpy as np
import cv2
from matplotlib import pyplot as plt

#Download files from https://drive.google.com/file/d/1XdZLvORnCnfpyBYflh15I58VQrQdVlUe/view?usp=sharing

im = cv2.imread("SkinDetection\SkinTrain1.jpg")
mask = cv2.imread("SkinDetection\SkinTrain1_mask.jpg",0)

im_hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
h = im_hsv[:,:,0]
s = im_hsv[:,:,1]

h_skin = h[mask >= 128]
s_skin = s[mask >= 128]
h_nonskin = h[mask < 128]
s_nonskin = s[mask < 128]


cv2.imshow('image',im)
cv2.imshow('mask',mask)
cv2.imshow('hue',h)
cv2.imshow('saturation',s)

plt.plot(h_nonskin,s_nonskin,'b.')
plt.plot(h_skin,s_skin,'r.')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
