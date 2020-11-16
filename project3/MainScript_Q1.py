# /--------------------------------------/
# NAME: Antonis Agapiou
# E-MAIL: cs141081@uniwa.gr
# A.M: 711141081
# CLASS: Computer Vision
# LAB: CV Lab Group 1
# /--------------------------------------/

import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth, AgglomerativeClustering, MiniBatchKMeans, OPTICS
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xlsxwriter

# ----- Initializing excel -------

workbook = xlsxwriter.Workbook("Results.xlsx")
worksheet = workbook.add_worksheet()
col = 0
row = 0
worksheet.write(row, col, 'Feature Extraction')
worksheet.write(row, col + 1, 'Clustering Technique')
worksheet.write(row, col + 2, 'Classification Technique')
worksheet.write(row, col + 3, 'Train Data Ratio')
worksheet.write(row, col + 4, 'Accuracy(tr)')
worksheet.write(row, col + 5, 'Precision(tr)')
worksheet.write(row, col + 6, 'Recall(tr)')
worksheet.write(row, col + 7, 'F1-score(tr)')
worksheet.write(row, col + 8, 'Accuracy(te)')
worksheet.write(row, col + 9, 'Precision(te)')
worksheet.write(row, col + 10, 'Recall(te)')
worksheet.write(row, col + 11, 'F1-score(te)')

row = row + 1


# Function to write to excel

def write_excel(feature_descr, cluster, classifier, percentage, acc_tr, pre_tr, recal_tr, f1_tr, acc_te, pre_te,
                recal_te, f1_te):
    global worksheet
    global row
    worksheet.write(row, col, feature_descr)
    worksheet.write(row, col + 1, cluster)
    worksheet.write(row, col + 2, classifier)
    worksheet.write(row, col + 3, str(percentage)+"%")
    worksheet.write(row, col + 4, round(acc_tr, 2))
    worksheet.write(row, col + 5, round(pre_tr, 2))
    worksheet.write(row, col + 6, round(recal_tr, 2))
    worksheet.write(row, col + 7, round(f1_tr, 2))
    worksheet.write(row, col + 8, round(acc_te, 2))
    worksheet.write(row, col + 9, round(pre_te, 2))
    worksheet.write(row, col + 10, round(recal_te, 2))
    worksheet.write(row, col + 11, round(f1_te, 2))
    row = row + 1


# ---------Data pre-processing functions---------

def load_images_from_folder(folder, ImageSize):
    images = {}
    for filename in os.listdir(folder):
        category = []
        path = folder + filename + "/"
        for monkey in os.listdir(path):
            img = cv2.imread(path + "/" + monkey)
            # print(' .. parsing image', cat)
            if img is not None:
                # grayscale it
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # resize it, if necessary
                img = cv2.resize(img, (ImageSize[0], ImageSize[1]))

                category.append(img)
        images[filename] = category
        print(' . Finished parsing images of class ' + filename)
    return images


# Using this function to split data according to split number
# so if number = 60, data will be split 60/40 etc

def split_data(all_images, split_number):
    trainImages = {}
    testImages = {}
    for key, values in all_images.items():
        category_train = []
        category_test = []
        for i in range(len(values)):
            if i < (round(split_number * len(values) / 100)):
                category_train.append(values[i])
            else:
                category_test.append(values[i])
        trainImages[key] = category_train
        testImages[key] = category_test
    return trainImages, testImages


# --------- Dictionary Learning functions ---------

# Extract features from images
def detector_features(images, detectorToUse):
    print(' . start detecting points and calculating features for a given image set')
    detector_vectors = {}
    descriptor_list = []

    for nameOfCategory, availableImages in images.items():
        features = []
        for img in availableImages:  # reminder: val
            kp, des = detectorToUse.detectAndCompute(img, None)
            descriptor_list.extend(des)
            features.append(des)
        detector_vectors[nameOfCategory] = features
    print(' . finished detecting points and calculating features for a given image set')
    return [descriptor_list, detector_vectors]  # be aware of the []! this is ONE output as a list


# Learn visual dictionary
def VisualWordsCreation(k, descriptor_list, clusterName):
    print(". Using " + clusterName)
    if clusterName == "KMeans":
        print(' . calculating central points for the existing feature values.')
        batchSize = np.ceil(descriptor_list.__len__() / 50).astype('int')
        kmeansModel = MiniBatchKMeans(n_clusters=k, batch_size=batchSize, verbose=0)
        kmeansModel.fit(descriptor_list)
        visualWords = kmeansModel.cluster_centers_  # a.k.a. centers of reference
        print(' . done calculating central points for the given feature set.')
        return visualWords, kmeansModel
    elif clusterName == "OPTICS":
        print(' . calculating central points for the existing feature values.')
        OPTICSModel = OPTICS()
        OPTICSModel.fit(descriptor_list)
        visualWords = OPTICSModel.cluster_centers_
        print(' . done calculating central points for the given feature set.')
        return visualWords, OPTICSModel


# --------- Encode functions ---------

def mapFeatureValsToHistogram(DataFeaturesByClass, visualWords, TrainedModel):
    # depending on the approach you may not need to use all inputs
    histogramsList = []
    targetClassList = []
    numberOfBinsPerHistogram = visualWords.shape[0]

    for categoryIdx, featureValues in DataFeaturesByClass.items():
        for tmpImageFeatures in featureValues:  # yes, we check one by one the values in each image for all images
            tmpImageHistogram = np.zeros(numberOfBinsPerHistogram)
            tmpIdx = list(TrainedModel.predict(tmpImageFeatures))
            clustervalue, visualWordMatchCounts = np.unique(tmpIdx, return_counts=True)
            tmpImageHistogram[clustervalue] = visualWordMatchCounts
            # do not forget to normalize the histogram values
            numberOfDetectedPointsInThisImage = tmpIdx.__len__()
            tmpImageHistogram = tmpImageHistogram / numberOfDetectedPointsInThisImage

            # now update the input and output coresponding lists
            histogramsList.append(tmpImageHistogram)
            targetClassList.append(categoryIdx)

    return histogramsList, targetClassList


# Loading images and labels
all_images_dir = 'C:/Users/green/Documents/Datasets/MonekySpecies/all_data/'


# defining the size of the image
inputImageSize = [200, 200, 3]

# dictionary with feature descriptors,clustering algorithms,classifier etc
feature_descriptors = {
    "SIFT": {
        "obj": cv2.xfeatures2d.SIFT_create()
    },
    "ORB": {
        "obj": cv2.ORB_create()
    },
    "BRISK": {
        "obj": cv2.BRISK_create()
    }
}

clustering_algorithms = ["KMeans"]
classifiers = {
    "SVM": {
        "obj": SVC()
    },
    "ΚNeighborsClassifier": {
        "obj": KNeighborsClassifier()
    },
    "Gaussian NB": {
        "obj": GaussianNB()
    }
}
train_test_percentages = [80, 60]

all_Images = load_images_from_folder(all_images_dir, inputImageSize)
# Main loop
for percentage in train_test_percentages:
    train_images, test_images = split_data(all_Images, percentage)
    for feature_descriptor in feature_descriptors:
        trainDataFeatures = detector_features(train_images, feature_descriptors[feature_descriptor]["obj"])
        TrainDescriptorList = trainDataFeatures[0]
        # Learn visual dictionary using clustering
        numberOfClasses = train_images.__len__()  # retrieve num of classes from dictionary
        possibleNumOfCentersToUse = 10 * numberOfClasses
        for cluster in clustering_algorithms:
            visualWords, TrainedModel = VisualWordsCreation(possibleNumOfCentersToUse, TrainDescriptorList, cluster)
            trainBoVWFeatureVals = trainDataFeatures[1]
            # create the train input train output format
            trainHistogramsList, trainTargetsList = mapFeatureValsToHistogram(trainBoVWFeatureVals, visualWords,
                                                                              TrainedModel)
            X_train = np.stack(trainHistogramsList, axis=0)
            # Convert Categorical Data for Scikit-Learn
            # Create a label (category) encoder object
            labelEncoder = preprocessing.LabelEncoder()
            labelEncoder.fit(trainTargetsList)
            # convert the categories from strings to names
            y_train = labelEncoder.transform(trainTargetsList)
            for classifier in classifiers:
                # ------------------ Training part ------------------------ #
                # KNeighbors classifier
                cls = classifiers[classifier]["obj"]
                cls.fit(X_train, y_train)
                print(percentage, " " + cluster + " " + feature_descriptor + " " + classifier)
                print("Accuracy of " + classifier + " training set: {:.2f}".format(cls.score(X_train, y_train)))

                print("Starting to test ")
                # calculate points and descriptor values per image
                testDataFeatures = detector_features(test_images, feature_descriptors[feature_descriptor]["obj"])
                # Takes the sift feature values that is seperated class by class for train data, we need this to
                # calculate the histograms
                testBoVWFeatureVals = testDataFeatures[1]
                # create the test input / test output format
                testHistogramsList, testTargetsList = mapFeatureValsToHistogram(testBoVWFeatureVals, visualWords,
                                                                                TrainedModel)
                X_test = np.array(testHistogramsList)
                y_test = labelEncoder.transform(testTargetsList)
                # classification tree
                # predict outcomes for test data and calculate the test scores
                y_pred_train = cls.predict(X_train)
                y_pred_test = cls.predict(X_test)
                # calculate the scores
                acc_train = accuracy_score(y_train, y_pred_train)
                acc_test = accuracy_score(y_test, y_pred_test)
                pre_train = precision_score(y_train, y_pred_train, average='macro')
                pre_test = precision_score(y_test, y_pred_test, average='macro')
                rec_train = recall_score(y_train, y_pred_train, average='macro')
                rec_test = recall_score(y_test, y_pred_test, average='macro')
                f1_train = f1_score(y_train, y_pred_train, average='macro')
                f1_test = f1_score(y_test, y_pred_test, average='macro')

                # print the scores
                print('Accuracy scores of' + classifier + ' classifier using ' + cluster + ' ' + feature_descriptor +
                      ' with percentage: ' + str(percentage) + ' are:',
                      'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
                print('Precision scores of SVM classifier are:',
                      'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
                print('Recall scores of SVM classifier are:',
                      'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
                print('F1 scores of SVM classifier are:',
                      'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))
                print('')
                write_excel(feature_descriptor, cluster, classifier, percentage, acc_train, pre_train, rec_train,
                            f1_train, acc_test, pre_test, rec_test, f1_test)
workbook.close()
