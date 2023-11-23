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
    cv2.moveWindow('mask',TARGET_SIZE[0],0)

    #############################################
    h, w = im_flipped.shape[:2]
    im_cropped = im_flipped[(int(h/2)-8):(int(h/2)+9),
                           (int(w/2)-8):(int(w/2)+9),
                           :]
    cv2.imshow('cropped', cv2.resize(im_cropped, (128,128)))
    cv2.moveWindow('cropped',0,TARGET_SIZE[1])
    b = int(np.mean(im_cropped[:,:,0]))
    g = int(np.mean(im_cropped[:,:,1]))
    r = int(np.mean(im_cropped[:,:,2]))

    cv2.rectangle(im_flipped,
                  (int(w/2)-8,int(h/2)-8),
                  (int(w/2)+8,int(h/2)+8),
                  (255,255,255))
    cv2.putText(im_flipped, str(b), (20, 30),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
    cv2.putText(im_flipped, str(g), (90, 30),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
    cv2.putText(im_flipped, str(r), (160, 30),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
    #############################################

    #print(np.sum(mask/255)/(h*w))

    if(np.sum(mask/255) > 0.01*h*w):
        cv2.putText(im_flipped,'Coke',(50,100),cv2.FONT_HERSHEY_PLAIN,5,(255,255,255))

    cv2.imshow('camera', im_flipped)
    cv2.moveWindow('camera',0,0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

