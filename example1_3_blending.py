import numpy as np
import cv2

TARGET_SIZE = (640,360)

cap = cv2.VideoCapture(0)

count = 1

while(True):
    ret,im = cap.read()
    im_resized = cv2.resize(im, TARGET_SIZE)

    if count > 5:
        im0 = im1
        im1 = im2
        im2 = im3
        im3 = im_flipped

        im_flipped = cv2.flip(im_resized, 1)
        im_out = (0.2*im0 + 0.2*im1 + 0.2*im2 + 0.2*im3 + 0.2*im_flipped).astype(np.uint8)
        cv2.imshow('camera',im_out)
    else:
        im0 = im1 = im2 = im3 = im_flipped = cv2.flip(im_resized, 1)
        cv2.imshow('camera',im_flipped)

    count = count+1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
