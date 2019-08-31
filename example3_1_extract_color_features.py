import numpy as np
import cv2
from matplotlib import pyplot as plt

im = cv2.imread("SkinTrain1.jpg")
mask = cv2.imread("SkinTrain1_mask.jpg",0)

im_hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
h = im_hsv[:,:,0]
s = im_hsv[:,:,1]

h_skin = [val for (i,val) in enumerate(h.reshape(1,-1)[0]) if mask.reshape(1,-1)[0][i] >= 128]
s_skin = [val for (i,val) in enumerate(s.reshape(1,-1)[0]) if mask.reshape(1,-1)[0][i] >= 128]
h_nonskin = [val for (i,val) in enumerate(h.reshape(1,-1)[0]) if mask.reshape(1,-1)[0][i] < 128]
s_nonskin = [val for (i,val) in enumerate(s.reshape(1,-1)[0]) if mask.reshape(1,-1)[0][i] < 128]

cv2.imshow('image',im)
cv2.imshow('mask',mask)
cv2.imshow('hue',h)
cv2.imshow('saturation',s)

plt.plot(h_nonskin,s_nonskin,'b.')
plt.plot(h_skin,s_skin,'r.')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
