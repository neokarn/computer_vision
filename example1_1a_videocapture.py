import cv2
print(cv2.__version__)

cap = cv2.VideoCapture(0)

#CAP_SIZE = (1280,720)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_SIZE[0])
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_SIZE[1])

ret,im = cap.read()

#print(im.shape)
#print(type(im))
#print(im[0,0])
#print(im[0,0,0])
#print(type(im[0,0,0]))

cv2.imshow('camera',im)
cv2.waitKey()
cap.release()
