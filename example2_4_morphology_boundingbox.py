import numpy as np
import cv2

cap = cv2.VideoCapture('C://Users//surface//Downloads//ExampleBGSubtraction.avi')

haveFrame,bg = cap.read()

while(cap.isOpened()):
    haveFrame,im = cap.read()

    if (not haveFrame) or (cv2.waitKey(1) & 0xFF == ord('q')):
        break

    diffc = cv2.absdiff(im,bg)
    diffg = cv2.cvtColor(diffc,cv2.COLOR_BGR2GRAY)
    bwmask = cv2.inRange(diffg,50,255)

    bwmask_median = cv2.medianBlur(bwmask,5)

    kernel = np.ones((55,25), np.uint8)
    bwmask_close = cv2.morphologyEx(bwmask_median, cv2.MORPH_CLOSE, kernel)

    contours,hierarchy = cv2.findContours(bwmask_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('bwmask_median', bwmask_median)
    cv2.moveWindow('bwmask_median',10,10)
    cv2.imshow('bwmask_close',bwmask_close)
    cv2.moveWindow('bwmask_close', 400, 10)
    cv2.imshow('im', im)
    cv2.moveWindow('im', 800, 10)

cap.release()
cv2.destroyAllWindows()
