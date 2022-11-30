import cv2
import numpy as np
from random import randint

TARGET_SIZE = (640,360)

cap = cv2.VideoCapture(0)

row = randint(20, TARGET_SIZE[1] - 20)
col = randint(20, TARGET_SIZE[0] - 20)

score = 0

while(True):
    ret,im = cap.read()
    im_resized = cv2.resize(im, TARGET_SIZE)
    im_flipped = cv2.flip(im_resized, 1)

    im_cropped = im_flipped[row-8:row+9,col-8:col+9,:]
    im_hsv = cv2.cvtColor(im_cropped, cv2.COLOR_BGR2HSV)

    hue = np.median(im_hsv[:,:,0])*2
    sat = np.median(im_hsv[:, :, 1])
    print(hue,sat)

    cv2.rectangle(im_flipped,
                  (col-8,row-8),
                  (col+8,row+8),
                  (255,0,255),
                  3)

    if hue > 300 and hue < 350 and sat > 100:
        score += 1
        row = randint(20, TARGET_SIZE[1] - 20)
        col = randint(20, TARGET_SIZE[0] - 20)

    cv2.putText(im_flipped, str(score), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255))
    cv2.imshow('camera', im_flipped)
    cv2.moveWindow('camera',0,0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
