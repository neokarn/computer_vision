import cv2

ref = cv2.imread('conan1.jpg',cv2.COLOR_BGR2GRAY)
target = cv2.imread('conan2.jpg',cv2.COLOR_BGR2GRAY)
h,w,_ = target.shape
target = cv2.resize(target,(int(w*0.5),int(h*0.5)))

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(ref,None)
kp2, des2 = orb.detectAndCompute(target,None)

print(str(len(kp1))+","+str(len(kp2)))
print(des1.shape)
print(des2.shape)
print(kp1[0].pt)
print(kp1[0].size)
print(kp1[0].angle)
print(des1[0])

bf = cv2.BFMatcher(cv2.NORM_HAMMING)

matches = bf.match(des1,des2)

matches = sorted(matches, key = lambda x:x.distance)

keypoint_ref = ref.copy()
cv2.drawKeypoints(ref,kp1,keypoint_ref,None,flags=4)
keypoint_target = target.copy()
cv2.drawKeypoints(target,kp2,keypoint_target,None,flags=4)

result = cv2.drawMatches(ref,kp1,target,kp2,matches[:30],None, flags=2)

cv2.imshow('keypoint1',keypoint_ref)
cv2.imshow('keypoint2',keypoint_target)
cv2.imshow('match',result)

cv2.waitKey()

cv2.destroyAllWindows()
