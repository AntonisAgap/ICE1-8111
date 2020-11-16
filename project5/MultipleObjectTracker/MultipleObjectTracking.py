from __future__ import print_function
import sys
import cv2
from random import randint
from math import sqrt
import xlsxwriter
import time
import matplotlib.pyplot as plt

# ----- Initializing excel -------

workbook = xlsxwriter.Workbook("Results.xlsx")
worksheet = workbook.add_worksheet()
col = 0
row = 0
worksheet.write(row, col, 'Technique name')
worksheet.write(row, col + 1, 'Frame Number')
worksheet.write(row, col + 2, 'Processing Time')
worksheet.write(row, col + 3, 'Centroid Change Box 1')
worksheet.write(row, col + 4, 'Centroid Change Box 2')

row = row + 1


# Function to write to excel

def write_excel(techniqueName, frameNumber, processingTime, centroidChangeBox1, centroidChangeBox2):
    global worksheet
    global row
    worksheet.write(row, col, techniqueName)
    worksheet.write(row, col + 1, frameNumber)
    worksheet.write(row, col + 2, round(processingTime, 2))
    worksheet.write(row, col + 3, round(centroidChangeBox1, 2))
    worksheet.write(row, col + 4, round(centroidChangeBox2, 2))
    row = row + 1


trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker


# Step 2: Read First Frame of a Video
# Set video to load
videoPath = "wolfChase.mp4"

# Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)

# Read first frame
success, frame = cap.read()
# quit if unable to read the video file
if not success:
    print('Failed to read video')
    sys.exit(1)

# Step 3: Locate Objects in the First Frame
## Select boxes
bboxes = []
colors = []

# OpenCV's selectROI function doesn't work for selecting multiple objects in Python
# So we will call this function in a loop till we are done selecting all objects
while True:
    # draw bounding boxes over objects
    # selectROI's default behaviour is to draw box starting from the center
    # when fromCenter is set to false, you can draw box starting from top left corner
    bbox = cv2.selectROI('MultiTracker', frame)
    bboxes.append(bbox)
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv2.waitKey(0) & 0xFF
    if (k == 113):  # q is pressed
        break
cv2.destroyAllWindows()

print('Selected bounding boxes {}'.format(bboxes))


# Create MultiTracker object
def useMultitracker(trackerType, bboxes, frame):
    multiTracker = cv2.MultiTracker_create()
    oldx = []
    oldy = []
    global distances0
    global distances1
    distance = [0, 0]
    # Initialize MultiTracker
    for i, bbox in enumerate(bboxes):
        oldx.append(int(bbox[0]) + (int(bbox[2]) // 2))
        oldy.append(int(bbox[1]) + (int(bbox[3]) // 2))

        multiTracker.add(createTrackerByName(trackerType), frame, bbox)

    frameCount = 0
    # Step 5: Update MultiTracker & Display Results
    # Process video and track objects

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frameCount = frameCount + 1
        # get updated location of objects in subsequent frames
        start_time = time.time()
        success, boxes = multiTracker.update(frame)
        end_time = time.time()
        calcTime = end_time - start_time

        # draw tracked objects
        # print("Frame No:", frameCount)
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            # calculating centroid
            x, y = (int(newbox[0]) + (int(newbox[2]) // 2)), (int(newbox[1]) + (int(newbox[3]) // 2)),
            # drawing bounding box
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
            # drawing centroid
            cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=3)

            cv2.putText(frame, "centroid box:" + str(i), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            distance[i] = sqrt((x - oldx[i]) ** 2 + (y - oldy[i]) ** 2)

            # print(". Box No:", i)
            # print(". Distance:", distance[i])
            # print(".", oldx[i], oldy[i])
            oldx[i], oldy[i] = x, y
            # print(".", oldx[i], oldy[i])

        write_excel(trackerType, frameCount, calcTime, distance[0], distance[1])
        distances0.append(distance[0])
        distances1.append(distance[1])
        # show frame
        cv2.imshow('MultiTracker: ' + trackerType, frame)

        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break


distances0 = []
distances1 = []

cap = cv2.VideoCapture(videoPath)
useMultitracker("CSRT", bboxes, frame)
plt.plot(distances0, 'r', distances1, 'b')
plt.ylabel("distances")
plt.legend(['red= Centroid0, blue= Centroid1'], loc='upper left')
plt.title("CSRT MultiTracker")
plt.show()
distances0.clear()
distances1.clear()

cap = cv2.VideoCapture(videoPath)
useMultitracker("KCF", bboxes, frame)
plt.plot(distances0, 'r', distances1, 'b')
plt.ylabel("distances")
plt.legend(['red= Centroid0, blue= Centroid1'], loc='upper left')
plt.title("KCF MultiTracker")
plt.show()
distances0.clear()
distances1.clear()

cap = cv2.VideoCapture(videoPath)
useMultitracker("BOOSTING", bboxes, frame)
plt.plot(distances0, 'r', distances1, 'b')
plt.ylabel("distances")
plt.xlabel("frames")
plt.legend(['red= Centroid0, blue= Centroid1'], loc='upper left')
plt.title("BOOSTING MultiTracker")
plt.show()

workbook.close()
