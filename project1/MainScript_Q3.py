# /--------------------------------------/
# NAME: Antonis Agapiou
# E-MAIL: cs141081@uniwa.gr
# A.M: 711141081
# CLASS: Computer Vision
# LAB: CV Lab Group 1
# /--------------------------------------/

from keras.datasets import mnist
import matplotlib.pyplot as plt
from random import sample
import numpy as np
import cv2
import skimage.metrics
import xlsxwriter

workbook = xlsxwriter.Workbook("ResultsQ3.xlsx")
worksheet = workbook.add_worksheet()
worksheet2 = workbook.add_worksheet()
worksheet3 = workbook.add_worksheet()


# function to transform to image to magnitude spectrum
# and apply fourier transformation
def Fourier_Transformation(img_GRAY):
    # now do the fourier stuff
    f = np.fft.fft2(img_GRAY)  # find Fourier Transform
    fshift = np.fft.fftshift(f)  # move zero frequency component (DC component) from top left to center

    # and calculate the magitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    magnitude_spectrum_img = np.round(magnitude_spectrum).astype('uint8')

    rows, cols = img_GRAY.shape
    crow, ccol = rows // 2, cols // 2
    fshift[crow - 30:crow + 31, ccol - 30:ccol + 31] = 0

    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.round(np.real(img_back)).astype('uint8')
    return magnitude_spectrum_img, img_back


# function to load mnist images
def load_mnist_images():
    # Loading necessary data

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_threes = x_train[y_train == 3]
    x_train_fives = x_train[y_train == 5]
    x_train_eights = x_train[y_train == 8]
    x_train_nines = x_train[y_train == 9]

    threes = sample(list(x_train_threes), 5)
    fives = sample(list(x_train_fives), 5)
    eights = sample(list(x_train_eights), 5)
    nines = sample(list(x_train_nines), 5)

    return threes, fives, eights, nines


# function to display images
def display_images(image_list, digit_name):
    for image in image_list:
        cv2.imshow("Digit: " + digit_name, image)
        cv2.waitKey()


# Loading data
random_threes, random_fives, random_eights, random_nines = load_mnist_images()
# Reformatting images to cv2 appropriate format
random_threes_cv = []
random_fives_cv = []
random_eights_cv = []
random_nines_cv = []

for i in range(5):
    random_threes_cv.append(cv2.resize(random_threes[i], (480, 360)))
    random_fives_cv.append(cv2.resize(random_fives[i], (480, 360)))
    random_eights_cv.append(cv2.resize(random_eights[i], (480, 360)))
    random_nines_cv.append(cv2.resize(random_nines[i], (480, 360)))

# Displaying images
display_images(random_threes_cv, "Three")
display_images(random_fives_cv, "Five")
display_images(random_eights_cv, "Eight")
display_images(random_nines_cv, "Nine")
cv2.destroyAllWindows()

# for every image we transform it and we save it in a list

all_images_cv_fourier = []
all_images_cv_frequency = []

all_images_cv = [*random_threes_cv, *random_fives_cv, *random_eights_cv, *random_nines_cv]
for image in all_images_cv:
    image_freq, image_fourier = Fourier_Transformation(image)
    all_images_cv_frequency.append(image_freq)
    all_images_cv_fourier.append(image_fourier)


# displaying images
cv2.imshow("Digit:3,spatial-frequency", all_images_cv_frequency[1])
cv2.waitKey()
cv2.imshow("Digit:3,spatial-frequency", all_images_cv_fourier[1])
cv2.waitKey()
cv2.imshow("Digit:5,spatial-frequency", all_images_cv_frequency[6])
cv2.waitKey()
cv2.imshow("Digit:5,spatial-frequency", all_images_cv_fourier[6])
cv2.waitKey()
cv2.imshow("Digit:8,spatial-frequency", all_images_cv_frequency[11])
cv2.waitKey()
cv2.imshow("Digit:8,spatial-frequency", all_images_cv_fourier[11])
cv2.waitKey()
cv2.imshow("Digit:9,spatial-frequency", all_images_cv_frequency[16])
cv2.waitKey()
cv2.imshow("Digit:9,spatial-frequency", all_images_cv_fourier[16])
cv2.waitKey()



for i in range(len(all_images_cv)):
    for j in range(len(all_images_cv)):
        # comparing all original images between them
        (score, _) = skimage.metrics.structural_similarity(all_images_cv[i], all_images_cv[j], full=True)
        worksheet.write(i, j, score)
        # comparing all images in frequency spectrum
        (score, _) = skimage.metrics.structural_similarity(all_images_cv_frequency[i], all_images_cv_frequency[j],
                                                           full=True)
        worksheet2.write(i, j, score)
        # comparing all images after fourier transformation
        (score, _) = skimage.metrics.structural_similarity(all_images_cv_fourier[i], all_images_cv_fourier[j],
                                                           full=True)
        worksheet3.write(i, j, score)

workbook.close()

