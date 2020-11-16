from YOLO.SupportingFunctions.YOLOV3Functions import make_yolov3_model
from YOLO.SupportingFunctions.LoadingData import WeightReader, checkFileExist, downloadWeights


def initialize():
    if checkFileExist():
        print(". Weights loaded...")
    else:
        downloadWeights()
        print(". Weights loaded...")

    # define the model
    model = make_yolov3_model()
    # load the model weights
    weight_reader = WeightReader('yolov3.weights')
    # set the model weights into the model
    weight_reader.load_weights(model)
    # save the model to file
    model.save('model.h5')
