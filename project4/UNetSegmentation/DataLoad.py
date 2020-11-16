import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from sklearn.model_selection import train_test_split

# Data downloaded from http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
path_leftmask_test = "C:/Users/green/Documents/Datasets/MontgomerySet/ManualMask/leftMask"
path_rightmask_test = "C:/Users/green/Documents/Datasets/MontgomerySet/ManualMask/rightMask"
path_cxr_test = "C:/Users/green/Documents/Datasets/MontgomerySet/CXR_png"

# Data downloaded from http://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip
path_cxr_train = "C:/Users/green/Documents/Datasets/CXR_png"
# Data downloaded from https://www.kaggle.com/yoctoman/shcxr-lung-mask
path_mask_train = "C:/Users/green/Documents/Datasets/mask"

# path_temp_output_images = "C:/Users/green/Desktop/output"
# if not os.path.exists(path_temp_output_images):
#     os.makedirs(path_temp_output_images)

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1


def add_colored_mask(image, mask_image):
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    mask_image = mask_image.astype(np.uint8)
    image = image.astype(np.uint8)
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)
    mask_coord = np.where(mask != [0, 0, 0])
    mask[mask_coord[0], mask_coord[1], :] = [255, 0, 0]
    ret = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
    return ret


def loadMontgomeryDataset():
    X = []
    y = []
    count = 1
    for image in os.listdir(path_leftmask_test):
        if (path_leftmask_test + "/" + image).endswith('png') and (path_rightmask_test + "/" + image).endswith('png') \
                and (path_cxr_test + "/" + image).endswith('png'):
            left_image = cv2.imread(path_leftmask_test + "/" + image, cv2.IMREAD_GRAYSCALE)
            right_image = cv2.imread(path_rightmask_test + "/" + image, cv2.IMREAD_GRAYSCALE)
            cxr_image = cv2.imread(path_cxr_test + "/" + image, cv2.IMREAD_GRAYSCALE)
            left_image = cv2.resize(left_image, (IMG_WIDTH, IMG_HEIGHT))
            right_image = cv2.resize(right_image, (IMG_WIDTH, IMG_HEIGHT))
            cxr_image = cv2.resize(cxr_image, (IMG_WIDTH, IMG_HEIGHT))
            gt_image = cv2.add(left_image, right_image)
            y.append(gt_image)
            X.append(cxr_image)
            print("\r. [Montgomery dataset] Loaded", count, "images", end='')
            count = count + 1

    i = random.choice(range(len(X)))
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(X[i], cmap="gray")
    axs[0].set_title('X-Ray Image')
    axs[1].imshow(y[i], cmap="gray")
    axs[1].set_title('Mask')
    axs[2].imshow(add_colored_mask(X[i], y[i]))
    axs[2].set_title('Merged')
    fig.suptitle("Montgomery dataset Image")
    plt.show()

    # normalizing images
    for i in range(len(X)):
        X[i] = X[i] / 255.0
        y[i] = y[i] / 255.0

    return X, y


# Shenzhen Hospital dataset
def loadShenzhenDataset():
    X = []
    y = []
    names = []
    # to dataset me tis maskes den exei oles tis maskes gia oles tis xrays giafto xrhsimopoiw aftes pou exei mono
    count = 0
    for image in os.listdir(path_mask_train):
        if (path_mask_train + "/" + image).endswith('png'):
            names.append(image[:-9] + ".png")
            gt_image = cv2.imread(path_mask_train + "/" + image, cv2.IMREAD_GRAYSCALE)
            gt_image = cv2.resize(gt_image, (IMG_WIDTH, IMG_HEIGHT))
            y.append(gt_image)
            print("\r. [Shenzhen Hospital dataset mask] Loaded", count + 1, "images", end='')
            count = count + 1

    count = 0
    for name in names:
        cxr_image = cv2.imread(path_cxr_train + "/" + name, cv2.IMREAD_GRAYSCALE)
        cxr_image = cv2.resize(cxr_image, (IMG_WIDTH, IMG_HEIGHT))
        X.append(cxr_image)
        print("\r. [Shenzhen Hospital dataset train] Loaded", count + 1, "images", end='')
        count = count + 1
    i = random.choice(range(len(X)))
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(X[i], cmap="gray")
    axs[0].set_title('X-Ray Image')
    axs[1].imshow(y[i], cmap="gray")
    axs[1].set_title('Mask')
    axs[2].imshow(add_colored_mask(X[i], y[i]))
    axs[2].set_title('Merged')
    fig.suptitle("Sheznhen Hospital dataset Image")
    plt.show()

    # normalizing images
    for i in range(len(X)):
        X[i] = X[i] / 255.0
        y[i] = y[i] / 255.0

    return X, y


def loadData():
    Xm, ym = loadMontgomeryDataset()
    Xs, ys = loadShenzhenDataset()
    X = np.concatenate((Xm, Xs), axis=0)
    y = np.concatenate((ym, ys), axis=0)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=12)
    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.5, random_state=12)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_train = np.expand_dims(X_train, axis=3)
    y_train = np.expand_dims(y_train, axis=3)
    X_val = np.expand_dims(X_val, axis=3)
    y_val = np.expand_dims(y_val, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    y_test = np.expand_dims(y_test, axis=3)

    return X_train, y_train, X_val, y_val, X_test, y_test
