import cv2
import numpy as np
cap = cv2.VideoCapture(0)
_, frame = cap.read()
frame = cv2.flip(frame,1)
prvs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

while(1):
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    current = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev=prvs,
                                        next=current,
                                        flow=None,
                                        pyr_scale=0.3,
                                        levels=8,
                                        winsize=10,
                                        iterations=3,
                                        poly_n=5,
                                        poly_sigma=1.2,
                                        flags=0)
    #More info on Farneback' method
    #http://www.diva-portal.org/smash/get/diva2:273847/FULLTEXT01.pdf

    flowim = np.zeros_like(frame)
    #flowim = frame.copy()
    step = 10
    for y in range(0, flow.shape[0]-1, step):
        for x in range(0, flow.shape[1] - 1, step):
            pt1 = (x, y)
            pt2 = (int(x+flow[y, x, 0]/2), int(y+flow[y, x, 1]/2))

            flowim = cv2.arrowedLine(flowim, pt1, pt2, (0, 255, 0), 1, tipLength=0.4)

    cv2.imshow('FlowArrow', flowim)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    prvs = current.copy()

cap.release()
cv2.destroyAllWindows()
