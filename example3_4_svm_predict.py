import numpy as np
import cv2

for im_id in range(1,4):
    print(im_id)
    im = cv2.imread("SkinTrain"+str(im_id)+".jpg")
    mask = cv2.imread("SkinTrain"+str(im_id)+"_mask.jpg",0)

    im_hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    h = im_hsv[:,:,0]
    s = im_hsv[:,:,1]

    h_skin = [val for (i, val) in enumerate(h.reshape(1, -1)[0]) if mask.reshape(1, -1)[0][i] >= 128]
    s_skin = [val for (i, val) in enumerate(s.reshape(1, -1)[0]) if mask.reshape(1, -1)[0][i] >= 128]
    h_nonskin = [val for (i, val) in enumerate(h.reshape(1, -1)[0]) if mask.reshape(1, -1)[0][i] < 128]
    s_nonskin = [val for (i, val) in enumerate(s.reshape(1, -1)[0]) if mask.reshape(1, -1)[0][i] < 128]

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

labels = np.zeros((len(h_skin_all) + len(h_nonskin_all), 1))
labels[0:len(h_skin_all)] = 1

features = np.zeros((len(h_skin_all) + len(h_nonskin_all), 2))
features[:, 0] = np.concatenate((h_skin_all, h_nonskin_all))
features[:, 1] = np.concatenate((s_skin_all, s_nonskin_all))


svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
#svm.setDegree(2)
svm.train(features.astype(np.float32), cv2.ml.ROW_SAMPLE, labels.astype(np.int32))

im = cv2.imread("SkinTest1.jpg")

im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
h = im_hsv[:, :, 0]
s = im_hsv[:, :, 1]

features = np.zeros((len(h.reshape(1,-1)[0]),2))
features[:,0] = h.reshape(1,-1)[0]
features[:,1] = s.reshape(1,-1)[0]

responses = svm.predict(features.astype(np.float32))[1]

cv2.imshow("skindetection",responses.reshape(np.shape(h)))

cv2.waitKey(0)
cv2.destroyAllWindows()

