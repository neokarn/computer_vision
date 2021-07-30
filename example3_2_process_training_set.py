import numpy as np
import cv2
from matplotlib import pyplot as plt

# Download files from https://drive.google.com/file/d/1XdZLvORnCnfpyBYflh15I58VQrQdVlUe/view?usp=sharing

for im_id in range(1, 4):
    print(im_id)
    im = cv2.imread("SkinTrain" + str(im_id) + ".jpg")
    mask = cv2.imread("SkinTrain" + str(im_id) + "_mask.jpg", 0)

    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h = im_hsv[:, :, 0]
    s = im_hsv[:, :, 1]

    h_skin = h[mask >= 128]
    s_skin = s[mask >= 128]
    h_nonskin = h[mask < 128]
    s_nonskin = s[mask < 128]

    if im_id == 1:
        h_skin_all = h_skin
        s_skin_all = s_skin
        h_nonskin_all = h_nonskin
        s_nonskin_all = s_nonskin
    else:
        h_skin_all = np.concatenate((h_skin_all, h_skin))
        s_skin_all = np.concatenate((s_skin_all, s_skin))
        h_nonskin_all = np.concatenate((h_nonskin_all, h_nonskin))
        s_nonskin_all = np.concatenate((s_nonskin_all, s_nonskin))

plt.figure(0)
plt.plot(h_nonskin_all, s_nonskin_all, 'b.')
plt.plot(h_skin_all, s_skin_all, 'r.')

plt.figure(1)
plt.subplot(2,1,1)
plt.hist(h_nonskin_all,bins=180,color='b')
plt.subplot(2,1,2)
plt.hist(h_skin_all,bins=180,color='r')

plt.figure(2)
plt.subplot(2,1,1)
plt.hist(s_nonskin_all,bins=180,color='b')
plt.subplot(2,1,2)
plt.hist(s_skin_all,bins=180,color='r')

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
