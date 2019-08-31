import numpy as np
import cv2
from matplotlib import pyplot as plt

h_skin_hist = 0
h_nonskin_hist = 0
s_skin_hist = 0
s_nonskin_hist = 0
for im_id in range(1,4):
    print(im_id)
    im = cv2.imread("SkinTrain"+str(im_id)+".jpg")
    mask = cv2.imread("SkinTrain"+str(im_id)+"_mask.jpg",0)
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h_skin_hist = h_skin_hist+cv2.calcHist([im_hsv], [0], mask, [256], [0, 256])
    h_nonskin_hist = h_nonskin_hist+cv2.calcHist([im_hsv], [0], 255-mask, [256], [0, 256])
    s_skin_hist = s_skin_hist+cv2.calcHist([im_hsv], [1], mask, [256], [0, 256])
    s_nonskin_hist = s_nonskin_hist+cv2.calcHist([im_hsv], [1], 255-mask, [256], [0, 256])

h_skin_prob = h_skin_hist/sum(h_skin_hist)
h_nonskin_prob = h_nonskin_hist/sum(h_nonskin_hist)
s_skin_prob = s_skin_hist/sum(s_skin_hist)
s_nonskin_prob = s_nonskin_hist/sum(s_nonskin_hist)
skin_prob = sum(h_skin_hist)/(sum(h_nonskin_hist)+sum(h_skin_hist))
nonskin_prob = 1-skin_prob


im = cv2.imread("SkinTest4.jpg")

im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

mask = np.zeros(im_hsv.shape[:2],np.uint8)
for i in range(0,mask.shape[0]):
    for j in range (0,mask.shape[1]):
        nb_skin_prob = h_skin_prob[im_hsv[i][j][0]]*s_skin_prob[im_hsv[i][j][1]]*skin_prob
        nb_nonskin_prob = h_nonskin_prob[im_hsv[i][j][0]] * s_nonskin_prob[im_hsv[i][j][1]]*nonskin_prob
        if nb_skin_prob > nb_nonskin_prob:
            mask[i][j] = 255

cv2.imshow("image",im)
cv2.imshow("skindetection",mask)

cv2.waitKey(0)
cv2.destroyAllWindows()

