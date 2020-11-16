import numpy as np
import cv2
import sys

video_path = 'cctv.mp4'

# read video file
cap = cv2.VideoCapture(video_path)


fgbg = cv2.createBackgroundSubtractorMOG2()

while (cap.isOpened):

    # if ret is true than no error with cap.isOpened
    ret, frame = cap.read()

    if ret == True:

        # apply background substraction
        blurframe = cv2.GaussianBlur(frame, (15, 15), 0)
        fgmask = fgbg.apply(blurframe)

        (im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # looping for contours
        for c in contours:
            if cv2.contourArea(c) < 500:
                continue

            # get bounding box from countour
            (x, y, w, h) = cv2.boundingRect(c)

            # draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)

        cv2.imshow('fgmask', fgmask)
        cv2.imshow('frame', frame)
        cv2.imshow('gaussianblur',blurframe)

        cv2.imwrite("fgmask.png", fgmask)
        cv2.imwrite("frame.png",frame)
        cv2.imwrite("gaussianblur.png",blurframe)
        cv2.waitKey()
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break
    else:
        if cv2.waitKey() & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
