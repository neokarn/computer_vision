import cv2

#Download images from
#https://drive.google.com/file/d/1TbVn2K3Kxtntne19vPOQjf5pGs6Jo72H/view?usp=sharing

ref = cv2.imread('conan1.jpg',cv2.COLOR_BGR2GRAY)
target = cv2.imread('conan2.jpg',cv2.COLOR_BGR2GRAY)
h,w,_ = target.shape
target = cv2.resize(target,(int(w*0.5),int(h*0.5)))

detector = cv2.SIFT_create()
matcher = cv2.BFMatcher()

#detector = cv2.ORB_create()
#matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

kp1, des1 = detector.detectAndCompute(ref,None)
kp2, des2 = detector.detectAndCompute(target,None)

print(str(len(kp1))+","+str(len(kp2)))
print(des1.shape)
print(des2.shape)
print(kp1[0].pt)
print(kp1[0].size)
print(kp1[0].angle)
print(des1[0])

matches = matcher.match(des1,des2)

matches = sorted(matches, key = lambda x:x.distance)

result = cv2.drawMatches(ref,kp1,target,kp2,matches[:50],None, flags=2)

keypoint_ref = ref.copy()
cv2.drawKeypoints(ref,kp1,keypoint_ref,None,flags=4)
keypoint_target = target.copy()
cv2.drawKeypoints(target,kp2,keypoint_target,None,flags=4)

cv2.imshow('keypoint1',keypoint_ref)
cv2.imshow('keypoint2',keypoint_target)
cv2.imshow('match',result)

cv2.waitKey()

cv2.destroyAllWindows()
