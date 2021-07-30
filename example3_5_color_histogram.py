import cv2
from matplotlib import pyplot as plt

# Download files from https://drive.google.com/file/d/1XdZLvORnCnfpyBYflh15I58VQrQdVlUe/view?usp=sharing

h_skin_hist = 0
h_nonskin_hist = 0
s_skin_hist = 0
s_nonskin_hist = 0
for im_id in range(1, 4):
    print(im_id)
    im = cv2.imread("SkinDetection\SkinTrain" + str(im_id) + ".jpg")
    mask = cv2.imread("SkinDetection\SkinTrain" + str(im_id) + "_mask.jpg", 0)
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h_skin_hist = h_skin_hist + cv2.calcHist([im_hsv], [0], mask, [256], [0, 256])
    h_nonskin_hist = h_nonskin_hist + cv2.calcHist([im_hsv], [0], 255 - mask, [256], [0, 256])
    s_skin_hist = s_skin_hist + cv2.calcHist([im_hsv], [1], mask, [256], [0, 256])
    s_nonskin_hist = s_nonskin_hist + cv2.calcHist([im_hsv], [1], 255 - mask, [256], [0, 256])

h_skin_prob = h_skin_hist / sum(h_skin_hist) # P(h|skin)
h_nonskin_prob = h_nonskin_hist / sum(h_nonskin_hist) # P(h|non-skin)
s_skin_prob = s_skin_hist / sum(s_skin_hist) # P(s|skin)
s_nonskin_prob = s_nonskin_hist / sum(s_nonskin_hist) # P(s|non-skin)

plt.plot(h_skin_prob, 'r', label = 'P(h|skin)')
plt.plot(h_nonskin_prob, 'b', label = 'P(h|non-skin)')
plt.legend()
plt.figure()
plt.plot(s_skin_prob, 'r', label = 'P(s|skin)')
plt.plot(s_nonskin_prob, 'b', label = 'P(s|non-skin)')
plt.show()
