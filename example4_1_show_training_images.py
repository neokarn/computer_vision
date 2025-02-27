import numpy as np
import cv2

#Download files form https://drive.google.com/file/d/1eliHaSv_fCBbkwy7n7SxDIFdJluPPzBz/view?usp=sharing

count = 1
charlist = "ABCDF"
for char_id in range(0,5):
    for im_id in range(1,6):
        im = cv2.imread("AtoF//"+charlist[char_id]+"//"+str(im_id)+".bmp",0)
        cv2.imshow(str(count),im)
        cv2.moveWindow(str(count), 60*im_id, 100*char_id)
        count = count+1

cv2.waitKey(0)
cv2.destroyAllWindows()

