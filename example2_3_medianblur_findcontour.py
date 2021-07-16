import cv2

#Download 'ExampleBGSubtraction.avi' from https://drive.google.com/file/d/1OD_A0wqN2Om2SusCztybu-_hMSUQuRt7/view?usp=sharing

cap = cv2.VideoCapture('ExampleBGSubtraction.avi')

haveFrame,bg = cap.read()

while(cap.isOpened()):
    haveFrame,im = cap.read()

    if (not haveFrame) or (cv2.waitKey(1) & 0xFF == ord('q')):
        break

    diffc = cv2.absdiff(im,bg)
    diffg = cv2.cvtColor(diffc,cv2.COLOR_BGR2GRAY)
    bwmask = cv2.inRange(diffg,50,255)

    bwmask_median = cv2.medianBlur(bwmask,5)

    contours,hierarchy = cv2.findContours(bwmask_median, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #for opencv 4.x.x
    #contourmask,contours,hierarchy = cv2.findContours(bwmask_median, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #for opencv 3.2.x 3.4.x
    #contourmask,contours,hierarchy = cv2.findContours(bwmask_median.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #for opencv 3.1.x

    im_out_contour = im.copy()
    cv2.drawContours(im_out_contour, contours, -1, (0, 255, 0), 1)

    im_out_boundingbox = im.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(im_out_boundingbox, (x, y), (x + w, y + h), (0, 0, 255), 2)


    cv2.imshow('bwmask', bwmask)
    cv2.moveWindow('bwmask',10,10)
    cv2.imshow('bwmask_median', bwmask_median)
    cv2.moveWindow('bwmask_median', 400, 10)
    cv2.imshow('im_out_contour', im_out_contour)
    cv2.moveWindow('im_out_contour', 10, 350)
    cv2.imshow('im_out_boundingbox', im_out_boundingbox)
    cv2.moveWindow('im_out_boundingbox', 400, 350)


cap.release()
cv2.destroyAllWindows()
