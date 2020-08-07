import cv2
import numpy as np

CAP_SIZE = (1280,720)
#TARGET_SIZE = (640,360)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_SIZE[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_SIZE[1])

while(True):
    ret,im = cap.read()    
    #im_resized = cv2.resize(im,TARGET_SIZE)
    #print(im_resized.shape)

    ############ Flipping ############################
    #im_flipped = cv2.flip(im_resized,1)
    ##################################################

    ############ Blurred ############################
    #L = 25
    #kernel = np.ones((L, L), np.float32) / L / L
    #im_blurred = cv2.filter2D(im_flipped, -1, kernel)
    ##################################################

    cv2.imshow('camera', im)#######################################

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
