import numpy as np
import cv2

TARGET_SIZE = (640,360)

cap = cv2.VideoCapture(0)

while(True):
    ret,im = cap.read()
    im_resized = cv2.resize(im, TARGET_SIZE)
    im_flipped = cv2.flip(im_resized, 1)

    h,w = im.shape[:2]

    mask = cv2.inRange(im_flipped,(0,0,90),(50,50,255))
    cv2.imshow('mask', mask)

    if(np.sum(mask) > 300000):
        cv2.putText(im_flipped,'Coke',(50,100),cv2.FONT_HERSHEY_PLAIN,5,(255,255,255))

    cv2.imshow('camera', im_flipped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
