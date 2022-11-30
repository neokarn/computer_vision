import numpy as np
import cv2

cap = cv2.VideoCapture(0)

L = 10

im_list = []
while(True):
    ret,im = cap.read()
    im_resized = cv2.resize(im, (640,360))
    im_flipped = cv2.flip(im_resized, 1).astype(np.float)

    if len(im_list) == L:
        im_list = im_list[1:]


    im_list.append(im_flipped)
    im_out = sum(im_list)/len(im_list)
    cv2.imshow('camera',im_out.astype(np.uint8))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
