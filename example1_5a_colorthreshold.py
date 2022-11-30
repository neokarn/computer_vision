import numpy as np
import cv2

cap = cv2.VideoCapture(0)

TARGET_SIZE = (640,360)

while(True):
    ret,im = cap.read()
    im_resized = cv2.resize(im, TARGET_SIZE)
    im_flipped = cv2.flip(im_resized, 1)    

    mask = cv2.inRange(im_flipped,(0,0,90),(50,50,255))
    cv2.imshow('mask', mask)
    cv2.moveWindow('mask',TARGET_SIZE[0],0)

    #print(np.sum(mask/255))

    if(np.sum(mask/255) > 10000):
        cv2.putText(im_flipped,'Coke',(50,100),cv2.FONT_HERSHEY_PLAIN,5,(255,255,255))

    cv2.imshow('camera', im_flipped)
    cv2.moveWindow('camera',0,0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
