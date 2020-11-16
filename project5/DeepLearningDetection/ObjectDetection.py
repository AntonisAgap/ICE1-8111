import cv2
import time
import numpy as np
import os
import sys
import requests


def downloadModel(file_name):
    link = "https://github.com/PINTO0309/MobileNet-SSD-RealSense/raw/master/caffemodel/MobileNetSSD" \
           "/MobileNetSSD_deploy.caffemodel"
    with open(file_name, "wb") as f:
        print("Downloading %s..." % file_name)
        response = requests.get(link, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                sys.stdout.flush()
            print(". File %s downloaded successfully!" % file_name)


def useSSD(filePath):
    caffeModel = "MobileNetSSD_deploy.caffemodel"

    if not os.path.exists(caffeModel):
        downloadModel(caffeModel)


    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt',
                                   'MobileNetSSD_deploy.caffemodel')

    print("[INFO] starting video stream...")
    cap = cv2.VideoCapture(filePath)

    timeSum = 0

    r, frame = cap.read()

    start_time = time.time()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    end_time = time.time()
    # print("Elapsed time: ", end_time - start_time)
    timeSum = timeSum + (end_time - start_time)
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.2:
            # extract the index of the class label
            idx = int(detections[0, 0, i, 1])
            # then compute the coords
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                                         confidence * 100)
            if "person" in label:
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                coords = (startX, startY, endX-startX, endY-startY)

            # show the output frame
            cv2.imshow("Frame", frame)
            cv2.imwrite("personSSD.png",frame)
            cv2.waitKey()
            return coords, timeSum

