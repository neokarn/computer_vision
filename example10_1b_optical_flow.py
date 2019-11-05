import cv2
cap = cv2.VideoCapture(0)
_, frame = cap.read()
frame = cv2.flip(frame,1)
prvs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

while(1):
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    current = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,current, None, 0.3, 8, 10, 3, 5, 1.2, 0)

    flowim = frame.copy()
    step = 15
    for y in range(0, flow.shape[0]-1, step):
        for x in range(0, flow.shape[1] - 1, step):
            pt1 = (x, y)
            pt2 = (int(x+flow[y, x, 0]/2), int(y+flow[y, x, 1]/2))

            if(abs(flow[y,x,0]) < 5 and abs(flow[y,x,1]) < 5):
                flowim = cv2.arrowedLine(flowim,pt1,pt2,(255,255,255),1,tipLength=0.4)
            elif(flow[y,x,0] < 0 and abs(flow[y,x,1]) <= abs(flow[y,x,0])):
                flowim = cv2.arrowedLine(flowim, pt1, pt2, (0, 0, 255), 2,tipLength=0.4)
            elif (flow[y, x, 0] > 0 and abs(flow[y, x, 1]) <= abs(flow[y, x, 0])):
                flowim = cv2.arrowedLine(flowim, pt1, pt2, (0, 255, 0), 2,tipLength=0.4)
            elif (flow[y, x, 1]  <0):
                flowim = cv2.arrowedLine(flowim, pt1, pt2, (0, 255, 255), 2,tipLength=0.4)
            elif (flow[y, x, 1] > 0):
                flowim = cv2.arrowedLine(flowim, pt1, pt2, (255, 100, 100), 2,tipLength=0.4)
            else:
                flowim = cv2.arrowedLine(flowim, pt1, pt2, (255, 255, 255), 1,tipLength=0.4)

    cv2.imshow('FlowArrow', flowim)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    prvs = current.copy()

cap.release()
cv2.destroyAllWindows()
