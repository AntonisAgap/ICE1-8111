import cv2
import time


def useHOG(filePath):
    global y
    print("[INFO] loading model...")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    print("[INFO] starting video stream...")
    cap = cv2.VideoCapture(filePath)

    timeSum = 0

    r, frame = cap.read()

    start_time = time.time()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    rects, weights = hog.detectMultiScale(gray_frame, winStride=(10, 10))
    # Measure elapsed time for detections
    end_time = time.time()
    # print("Elapsed time:", end_time - start_time)
    timeSum = timeSum + (end_time - start_time)

    for i, (x, y, w, h) in enumerate(rects):
        if weights[i] < 0.5:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        coords = (x, y, w, h)
        cv2.putText(frame, "person" + str(weights[i]), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.imshow("preview", frame)
    cv2.imwrite("personHOG.png", frame)
    cv2.waitKey()
    return coords, timeSum
