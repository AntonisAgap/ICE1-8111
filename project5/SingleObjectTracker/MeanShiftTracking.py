import numpy as np
import cv2
from Project_5.HOG_ObjectDetection.ObjectDetection import useHOG
from Project_5.DeepLearningDetection.ObjectDetection import useSSD
import time

coords, duration = useSSD("spectre.mp4")
# coords, duration = useHOG("spectre.mp4")

# setup initial location of window # you may play here with various techniques
# on how to initially get the position
c, r, w, h = coords[0], coords[1], coords[2], coords[3]
track_window = (c, r, w, h)

cap = cv2.VideoCapture('spectre.mp4')

# take first frame of the video
ret, frame = cap.read()
# set up the ROI for tracking
roi = frame[r:r + w, c:c + h]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

sum = 0
frameCount = 0

while (1):
    ret, frame = cap.read()

    if ret == True:
        frameCount = frameCount + 1

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)


        start_time = time.time()
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        end_time = time.time()
        calcDuration = end_time - start_time
        sum = sum + calcDuration

        # Draw it on image
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv2.imshow('img2', img2)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()
print("Initial Calculation Time:",duration)
print("Average Calculation Time:",sum/frameCount)
