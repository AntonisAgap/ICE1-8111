import os  # dealing with directories
import matplotlib.pyplot as plt
import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder
import xlsxwriter


def init_excel():
    # ----- Initializing excel -------
    workbook = xlsxwriter.Workbook("Results.xlsx")
    worksheet = workbook.add_worksheet()
    col = 0
    row = 2
    worksheet.write(row, col, 'Technique name')
    worksheet.write(row, col + 1, 'Train Data ratio')
    worksheet.write(row, col + 2, 'Accuracy (tr)')
    worksheet.write(row, col + 3, 'Precision (tr)')
    worksheet.write(row, col + 4, 'Recall (tr)')
    worksheet.write(row, col + 5, 'F1 score (tr)')
    worksheet.write(row, col + 6, 'Accuracy (te)')
    worksheet.write(row, col + 7, 'Precision (te)')
    worksheet.write(row, col + 8, 'Recall (te)')
    worksheet.write(row, col + 9, 'F1 score (te)')

    row = row + 1
    return workbook, worksheet, col, row


def write_excel(technique, ratio, acc_tr, pre_tr, recal_tr, f1_tr, acc_te, pre_te, recal_te, f1_te,
                worksheet, row, col):
    worksheet.write(row, col, technique)
    worksheet.write(row, col + 1, ratio + "%")
    worksheet.write(row, col + 2, round(acc_tr, 2))
    worksheet.write(row, col + 3, round(pre_tr, 2))
    worksheet.write(row, col + 4, round(recal_tr, 2))
    worksheet.write(row, col + 5, round(f1_tr, 2))
    worksheet.write(row, col + 6, round(acc_te, 2))
    worksheet.write(row, col + 7, round(pre_te, 2))
    worksheet.write(row, col + 8, round(recal_te, 2))
    worksheet.write(row, col + 9, round(f1_te, 2))


# Data from https://www.kaggle.com/slothkong/10-monkey-species/
# Defining paths
training_data = 'C:/Users/green/Documents/Datasets/MonkeySpeciesOriginal/training/training/'
testing_data = 'C:/Users/green/Documents/Datasets/MonkeySpeciesOriginal/validation/validation/'
labels_path = 'C:/Users/green/Documents/Datasets/MonkeySpeciesOriginal/monkey_labels.txt'

# Getting info for the names & labels
cols = ['Label', 'Latin Name', 'Common Name', 'Train Images', 'Validation Images']
labels_info = pd.read_csv(labels_path, names=cols, skiprows=1)

labels_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
common_names = []

for i in range(len(labels_info["Common Name"].tolist())):
    common_names.append(labels_info["Common Name"].tolist()[i].strip())

# map labels to common names
names_dict = dict(zip(labels_names, common_names))
print(names_dict)

# defining number of class and input Image Size

num_classes = 10
inputImageSize = [256, 256, 3]


# Loading function for the 10 monkey species dataset

def load_images_from_folder(folder, ImageSize):
    images = []
    labels = []
    for filename in os.listdir(folder):
        path = folder + filename + "/"
        for monkey in os.listdir(path):
            img = cv2.imread(path + "/" + monkey)
            if img is not None:
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (ImageSize[0], ImageSize[1]))
                images.append(img)
                labels.append(filename)
    print(' . Finished parsing images of folder: ' + folder + '.')
    return images, labels


print(". Loading train data...")
X_train, y_train = load_images_from_folder(training_data, inputImageSize)
print(". Train data loaded successfully!")

print(". Loading test data...")
X_test, y_test = load_images_from_folder(testing_data, inputImageSize)
print(". Test data loaded successfully!")

# We need to concatenate the lists so we can split them accordingly
X = [*X_train, *X_test]
y = [*y_train, *y_test]

# 0.2 = 80/20, 0.4 = 60/40
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Encoding labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.fit_transform(y_val)
y_test = label_encoder.fit_transform(y_test)

# Plotting a random image from every class
got_classes = []
got_images = []
while len(got_classes) != num_classes:
    i = random.choice(range(len(X_train)))
    if y_train[i] not in got_classes:
        got_classes.append(y_train[i])
        got_images.append(X_train[i])

# Test if everything is ok
plt.figure(figsize=(10, 5))
k = 0
for i in range(2):
    for j in range(5):
        ax = plt.subplot2grid((2, 5), (i, j))
        ax.imshow(got_images[k])
        ax.set_title("n" + str(got_classes[k]) + ": " + names_dict.get(got_classes[k]), fontsize=8)
        k = k + 1
plt.show()
