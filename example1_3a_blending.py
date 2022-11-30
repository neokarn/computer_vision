import numpy as np
import cv2

cap = cv2.VideoCapture(0)

ret,im = cap.read()
im_resized = cv2.resize(im, (640,360))
im_flipped = cv2.flip(im_resized, 1)
im0 = im1 = im2 = im3 = im_flipped

while(True):
    im0 = im1
    im1 = im2
    im2 = im3
    im3 = im_flipped
    
    ret,im = cap.read()
    im_resized = cv2.resize(im, (640,360))
    im_flipped = cv2.flip(im_resized, 1)

    im_out = (0.2*im0 + 0.2*im1 + 0.2*im2 + 0.2*im3 + 0.2*im_flipped).astype(np.uint8)
    cv2.imshow('camera',im_out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
