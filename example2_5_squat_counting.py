#Download videos from https://drive.google.com/file/d/1TNwCB5o3-joGgclbXkI01f2r4FazMf-w/view?usp=sharing

import cv2
import numpy as np
import matplotlib.pyplot as plt

def squatCounting(filename):

    cap = cv2.VideoCapture(filename)
    haveFrame, bg = cap.read()
    t = 0

    all_boundingboxes = []

    while (cap.isOpened()):
        haveFrame, im = cap.read()

        if (not haveFrame) or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        diffc = cv2.absdiff(im, bg)
        diffg = cv2.cvtColor(diffc, cv2.COLOR_BGR2GRAY)
        bwmask = cv2.inRange(diffg, 35, 255)

        bwmask_median = cv2.medianBlur(bwmask, 9)

        kernel = np.ones((7, 7), np.uint8)
        bwmask_open = cv2.morphologyEx(bwmask_median, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((75, 15), np.uint8)
        bwmask_close = cv2.morphologyEx(bwmask_open, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(bwmask_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        im_out_boundingbox = im.copy()
        i = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w*h > diffg.shape[0]*diffg.shape[1]*0.015 and w*h < diffg.shape[0]*diffg.shape[1]*0.2 \
                    and x > 2 and y > 2 and x+w < diffg.shape[1]-3 and y+h < diffg.shape[0]-3\
                    and h/w < 10:
                cv2.rectangle(im_out_boundingbox, (x, y), (x + w, y + h), (0, 0, 255), 2)
                all_boundingboxes.append([t, i , x, y, w, h])
                i = i+1

        cv2.putText(im_out_boundingbox, str(t), (30, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255))
        t = t+1

        cv2.imshow('bwmask', bwmask)
        cv2.moveWindow('bwmask', 10, 10)
        cv2.imshow('bwmask_close', bwmask_close)
        cv2.moveWindow('bwmask_close', 750, 10)
        cv2.imshow('im_out_boundingbox', im_out_boundingbox)
        cv2.moveWindow('im_out_boundingbox', 380, 350)

    all_boundingboxes = np.array(all_boundingboxes)
    n = max(all_boundingboxes[:,1])+1
    print(all_boundingboxes)
    print(n)

    for i in range(n):
        idx = np.where(all_boundingboxes[:,1] == i)[0]
        plt.subplot(n,1,i+1)
        plt.plot(all_boundingboxes[idx, 0],all_boundingboxes[idx, 5])
        #temp = np.expand_dims(all_boundingboxes[idx,5],axis = -1)
        #temp = cv2.medianBlur(temp.astype('uint8'), 5)
        #all_boundingboxes[idx, 5] = temp[:,0].astype('float')
        #plt.plot(all_boundingboxes[idx, 0],all_boundingboxes[idx,5])
    plt.show()

    cap.release()
    cv2.destroyAllWindows()

    return [] 


print(squatCounting('.\SquatCounting\Squat1_8_9.avi')) #Perfect Answer = [8 9]
print(squatCounting('.\SquatCounting\Squat2_16_17.avi')) #Perfect Answer = [16 17]
print(squatCounting('.\SquatCounting\Squat3_11_9_10.avi')) #Perfect Answer = [11 9 10]
