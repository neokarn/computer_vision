import numpy as np
import cv2
from skimage.feature import local_binary_pattern

#download images form https://drive.google.com/file/d/1JfJYr-qJvgt1Jyz-Gop-oRci-TipuQmb/view?usp=sharing


count = 0
class_list = ["Beef", "Omelet", "Spaghetti"]

label_train = np.zeros((15*3,1))
features_train = np.zeros((15*3,256))
for class_id in range(0,3):
    for im_id in range(1,16):
        im = cv2.imread("TextureClassification//"+class_list[class_id]+"//"+str(im_id)+".jpg")
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.resize(im_gray, (50, 50))
        lbp = local_binary_pattern(im_gray, 8, 2).astype(np.uint8)
       # cv2.imshow("lbp"+str(count),cv2.resize(lbp,(300,300)))
       # cv2.waitKey()
        h = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        h = h/sum(h)
        features_train[count] = h.reshape(1,-1)
        label_train[count] = class_id
        count = count+1

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(features_train.astype(np.float32), cv2.ml.ROW_SAMPLE,label_train.astype(np.int32))

for im_id in range(1,16):
    im = cv2.imread("TextureClassification//Unknown//" + str(im_id) + ".jpg")
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.resize(im_gray, (50, 50))
    lbp = local_binary_pattern(im_gray, 8, 2).astype(np.uint8)
    h = cv2.calcHist([lbp], [0], None, [256], [0, 256])
    h = h / sum(h)
    result = svm.predict(h.reshape(1,-1).astype(np.float32))[1]
    cv2.putText(im, class_list[result[0][0].astype(int)], (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
    cv2.imshow(str(im_id)+"="+class_list[result[0][0].astype(int)],im)
    cv2.moveWindow(str(im_id)+"="+class_list[result[0][0].astype(int)],100+((im_id-1)%5)*200,np.floor((im_id-1)/5).astype(int)*200)

cv2.waitKey(0)
cv2.destroyAllWindows()

