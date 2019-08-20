import numpy as np
import cv2

TARGET_SIZE = (640,360)

cap = cv2.VideoCapture(0)

while(True):
    ret,im = cap.read()
    im_resized = cv2.resize(im, TARGET_SIZE)
    im_flipped = cv2.flip(im_resized, 1)

    mask = cv2.inRange(im_flipped,(0,0,90),(50,50,255))
    cv2.imshow('mask', mask)

    #############################################
    h, w = im_flipped.shape[:2]
    b = im_flipped[int(h/2),int(w/2),0]
    g = im_flipped[int(h/2),int(w/2),1]
    r = im_flipped[int(h/2),int(w/2),2]

    cv2.putText(im_flipped, 'x', (int(w/2), int(h/2)),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    cv2.putText(im_flipped, str(b), (20, 30),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
    cv2.putText(im_flipped, str(g), (90, 30),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
    cv2.putText(im_flipped, str(r), (160, 30),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
    #############################################

    if(np.sum(mask) > 0.1*h*w):
        cv2.putText(im_flipped,'Coke',(50,100),cv2.FONT_HERSHEY_PLAIN,5,(255,255,255))

    cv2.imshow('camera', im_flipped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
