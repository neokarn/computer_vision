import cv2
import numpy as np
cap = cv2.VideoCapture(0)
_, frame = cap.read()
frame = cv2.flip(frame,1)
prvs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
flowhsv = np.zeros_like(frame)
flowhsv[...,1] = 255


while(1):
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    current = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,current, None, 0.3, 8, 10, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    flowhsv[...,0] = ang*180/np.pi/2
   # flowhsv[...,2] = 255
   # flowhsv[...,2] = 255*(mag > 5)
   # flowhsv[...,2] = mag*4
    flowhsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    flowim = cv2.cvtColor(flowhsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('FlowMagnitude', 4*mag.astype('uint8'))
    cv2.imshow('FlowImage',flowim)
    #cv2.imshow('Frame', (flowim/2+frame/2).astype('uint8'))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    prvs = current.copy()

cap.release()
cv2.destroyAllWindows()
