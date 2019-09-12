import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops


count = 0
class_list = ["Beef", "Omelet", "Spaghetti"]
L = 10
label_train = np.zeros((15*3,1))
features_train = np.zeros((15*3,4 * L * 3))
for class_id in range(0,3):
    for im_id in range(1,16):
        im = cv2.imread("Texture//"+class_list[class_id]+"//"+str(im_id)+".jpg")
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.resize(im_gray, (50, 50))
        im_gray = (im_gray / 16).astype(np.uint8)
        glcm = greycomatrix(im_gray, range(1, L + 1), [0, np.pi / 4, np.pi / 2], 16, symmetric=True, normed=True)
        glcm_props = np.zeros(4 * L * 3)
        glcm_props[0:(L * 3)] = greycoprops(glcm, 'ASM').reshape(1, -1)[0]
        glcm_props[(L * 3):(L * 3 * 2)] = greycoprops(glcm, 'contrast').reshape(1, -1)[0]
        glcm_props[(L * 3 * 2):(L * 3 * 3)] = greycoprops(glcm, 'homogeneity').reshape(1, -1)[0]
        glcm_props[(L * 3 * 3):(L * 3 * 4)] = greycoprops(glcm, 'correlation').reshape(1, -1)[0]
        features_train[count] = glcm_props
        label_train[count] = class_id
        count = count+1

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(features_train.astype(np.float32), cv2.ml.ROW_SAMPLE,label_train.astype(np.int32))

for im_id in range(1,16):
    im = cv2.imread("Texture//Unknown//" + str(im_id) + ".jpg")
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.resize(im_gray, (50, 50))
    im_gray = (im_gray / 16).astype(np.uint8)
    glcm = greycomatrix(im_gray, range(1, L + 1), [0, np.pi / 4, np.pi / 2], 16, symmetric=True, normed=True)
    glcm_props = np.zeros(4 * L * 3)
    glcm_props[0:(L * 3)] = greycoprops(glcm, 'ASM').reshape(1, -1)[0]
    glcm_props[(L * 3):(L * 3 * 2)] = greycoprops(glcm, 'contrast').reshape(1, -1)[0]
    glcm_props[(L * 3 * 2):(L * 3 * 3)] = greycoprops(glcm, 'homogeneity').reshape(1, -1)[0]
    glcm_props[(L * 3 * 3):(L * 3 * 4)] = greycoprops(glcm, 'correlation').reshape(1, -1)[0]
    result = svm.predict(glcm_props.reshape(1,-1).astype(np.float32))[1]
    cv2.putText(im, class_list[result[0][0].astype(int)], (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
    cv2.imshow(str(im_id)+"="+class_list[result[0][0].astype(int)],im)
    cv2.moveWindow(str(im_id)+"="+class_list[result[0][0].astype(int)],100+((im_id-1)%5)*200,np.floor((im_id-1)/5).astype(int)*200)

cv2.waitKey(0)
cv2.destroyAllWindows()

