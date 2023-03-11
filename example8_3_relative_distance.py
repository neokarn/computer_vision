import numpy as np
import cv2

cap = cv2.VideoCapture(0)

detector = cv2.SIFT_create()
matcher = cv2.BFMatcher()

#detector = cv2.ORB_create()
#matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

ref = cv2.imread('conan1.jpg', cv2.COLOR_BGR2GRAY)
h,w,_ = ref.shape
ref = cv2.resize(ref,(int(w*1.0),int(h*1.0)))

kp1, des1 = detector.detectAndCompute(ref, None)

while(True):
    _,target = cap.read()

    kp2, des2 = detector.detectAndCompute(target, None)

    matches = matcher.knnMatch(des1, des2, k=2) #Relative Matching

    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append(m)

    result = cv2.drawMatches(ref, kp1, target, kp2, good, None, flags=2)

    cv2.imshow('result', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
