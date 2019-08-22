import cv2

cap = cv2.VideoCapture('ExampleBGSubtraction.avi')

_,bg = cap.read()

while(cap.isOpened()):
    haveFrame,im = cap.read()

    if (not haveFrame) or (cv2.waitKey(1) & 0xFF == ord('q')):
        break

    diffc = cv2.absdiff(im,bg)
    diffg = cv2.cvtColor(diffc,cv2.COLOR_BGR2GRAY)
    bwmask = cv2.inRange(diffg,50,255)

    bwmask = cv2.medianBlur(bwmask,5)

    temp = bwmask.copy()
    contourmask,contours,hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(im, contours, -1, (0, 255, 0), 2)

    cv2.imshow('bwmask', bwmask)
    cv2.moveWindow('bwmask',10,10)
    cv2.imshow('contourmask',contourmask)
    cv2.moveWindow('contourmask', 400, 10)
    cv2.imshow('im', im)
    cv2.moveWindow('im', 800, 10)

cap.release()
cv2.destroyAllWindows()
