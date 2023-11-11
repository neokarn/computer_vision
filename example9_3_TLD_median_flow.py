import cv2

# For more info:
# https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

tracker = cv2.Tracker_create('MEDIANFLOW') #TLD MEDIANFLOW
#tracker = cv2.TrackerTLD_create()
#tracker = cv2.TrackerMedianFlow_create()
cap = cv2.VideoCapture(0)
_, frame = cap.read()
frame = cv2.flip(frame,1)
bbox = cv2.selectROI(frame, False)
tracker.init(frame, bbox)

while(1):
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    _, bbox = tracker.update(frame)
    pt1 = (int(bbox[0]), int(bbox[1]))
    pt2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2, 1)

    cv2.imshow('Tracking Result', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
