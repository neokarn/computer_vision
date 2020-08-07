import numpy as np
import cv2

CAP_SIZE = (1280,720)
TARGET_SIZE = (640,360)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_SIZE[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_SIZE[1])

while(True):
    ret,im = cap.read()
    im_resized = cv2.resize(im, TARGET_SIZE)
    im_flipped = cv2.flip(im_resized, 1)

    im_gray = cv2.cvtColor(im_flipped,cv2.COLOR_BGR2GRAY)
    im_hsv = cv2.cvtColor(im_flipped, cv2.COLOR_BGR2HSV)
    hue = im_hsv[:,:,0]
    #hue = cv2.applyColorMap((hue*(255/179)).astype('uint8'),cv2.COLORMAP_HSV)
    sat = im_hsv[:,:,1]
    val = im_hsv[:,:,2]

    cv2.imshow('Gray', im_gray)
    cv2.imshow('Hue', hue)
    cv2.imshow('Sat', sat)
    cv2.imshow('Val', val)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
