from os import path

from keras.models import load_model

from YOLO.DataLoadForObjectDetection import input_h, input_w, images, images_h, images_w
from YOLO.DataLoadForObjectDetection import photo_filenames
from YOLO.SupportingFunctions.DrawingFunctions import decode_netout, correct_yolo_boxes, do_nms, get_boxes, draw_boxes
from YOLO.SupportingFunctions.LoadingData import getCOCOLabels
from YOLO.ModelInitializationAndTraining import initialize

print(". Checking if file: model.h5 exist...")
if path.exists("model.h5"):
    print(". File: model.h5 exist...")
else:
    print(". File: model.h5 doesn't exist...")
    initialize()



def useModel(image, image_h, image_w, photo_filename):
    # loading model
    model = load_model("model.h5", compile=False)
    # loading labels
    labels = getCOCOLabels()
    # making prediction
    yhat = model.predict(image)
    # summarize the shape of the list of arrays

    anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    # define the probability threshold for detected objects
    class_threshold = 0.6
    boxes = list()
    for i in range(len(yhat)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)

    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    # suppress non-maximal boxes
    do_nms(boxes, 0.5)

    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

    # summarize what we found
    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])

    draw_boxes(photo_filename, v_boxes, v_labels, v_scores)


for i in range(len(images)):
    useModel(images[i], images_h[i], images_w[i], photo_filenames[i])
