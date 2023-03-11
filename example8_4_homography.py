import numpy as np
import cv2

cap = cv2.VideoCapture(0)

detector = cv2.SIFT_create()
matcher = cv2.BFMatcher()

#detector = cv2.ORB_create()
#matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

ref = cv2.imread('conan1.jpg', cv2.COLOR_BGR2GRAY)
h,w,_ = ref.shape
ref = cv2.resize(ref,(int(w*1.2),int(h*1.2)))

kp1, des1 = detector.detectAndCompute(ref, None)

while (True):
    _, target = cap.read()

    kp2, des2 = detector.detectAndCompute(target, None)

    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > 18:
        print(len(good))
        ref_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        target_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(ref_pts, target_pts, cv2.RANSAC, 20.0)
        matchesMask = mask.ravel().tolist()

        h, w, _ = ref.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        target = cv2.polylines(target, [np.int32(dst)], True, (0,0,255), 3, cv2.LINE_AA)

    result = cv2.drawMatches(ref, kp1, target, kp2, good, None, flags=2)

    cv2.imshow('result', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
