import numpy as np
import cv2

#Download files form https://drive.google.com/file/d/1eliHaSv_fCBbkwy7n7SxDIFdJluPPzBz/view?usp=sharing

count = 0
charlist = "ABCDF"
answerlist = "AAAAABBBBBCCCCCDDDDDFFFFF"

hog = cv2.HOGDescriptor((50,50),(50,50),(50,50),(50,50),9)
#hog = cv2.HOGDescriptor((50,50),(20,20),(10,10),(10,10),9)
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

knn = cv2.ml.KNearest_create()
knn.train(features_train.astype(np.float32),cv2.ml.ROW_SAMPLE,label_train.astype(np.int32))

for im_id in range(1,26):
    im = cv2.imread("AtoF//Unknown//" + str(im_id) + ".bmp", 0)

    im = cv2.resize(im, (50, 50))
    im = cv2.GaussianBlur(im, (3, 3), 0)
    h = hog.compute(im)
    _,result,_,_ = knn.findNearest(h.reshape(1,-1).astype(np.float32),1)
    im = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    if answerlist[im_id-1] != charlist[result[0][0].astype(int)]:
        im[:,:,2] = 255
        cv2.putText(im, charlist[result[0][0].astype(int)] , (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
    cv2.imshow(str(im_id) + "=" + charlist[result[0][0].astype(int)], cv2.resize(im, (100, 100)))
    cv2.moveWindow(str(im_id) + "=" + charlist[result[0][0].astype(int)], 100 + ((im_id - 1) % 5) * 120, np.floor((im_id - 1) / 5).astype(int) * 150)

cv2.waitKey(0)
cv2.destroyAllWindows()

