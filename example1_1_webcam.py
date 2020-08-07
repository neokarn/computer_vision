import cv2

#CAP_SIZE = (1280,720)

cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_SIZE[0])
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_SIZE[1])

while(True):
    ret,im = cap.read()

    #print(im.shape)

    cv2.imshow('camera',im) #uint8

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
