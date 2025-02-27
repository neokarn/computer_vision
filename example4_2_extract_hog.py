import numpy as np
import matplotlib.pyplot as plt
import cv2

#Download files form https://drive.google.com/file/d/1eliHaSv_fCBbkwy7n7SxDIFdJluPPzBz/view?usp=sharing

count = 0
charlist = "ABCDF"
colorlist = ["red","green","blue","olive","cyan"]

hog = cv2.HOGDescriptor((50,50),(50,50),(50,50),(50,50),9)
#WinSize, BlockSize, BlockStride, CellSize, NBins

label_train = np.zeros((25,1))

for char_id in range(0,5):
    for im_id in range(1,6):
        im = cv2.imread("AtoF//"+charlist[char_id]+"//"+str(im_id)+".bmp",0)

        im = cv2.resize(im, (50, 50))
        im = cv2.GaussianBlur(im, (3, 3), 0)
        h = hog.compute(im)

        if count == 0:
            features_train = h.reshape(1,-1)
        else:
            features_train = np.concatenate((features_train,h.reshape(1,-1)),axis = 0)

        label_train[count] = char_id
        count = count+1
        plt.figure(char_id)
        plt.plot(h, color=colorlist[char_id])
        plt.ylim(0,1)
        plt.figure(5)
        plt.plot(h,color=colorlist[char_id])
        plt.ylim(0, 1)

print(features_train)
print(label_train)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

