# /--------------------------------------/
# NAME: Antonis Agapiou
# E-MAIL: cs141081@uniwa.gr
# A.M: 711141081
# CLASS: Computer Vision
# LAB: CV Lab Group 1
# /--------------------------------------/

# required packages
import cv2
import os
import random
import time
import skimage.metrics
import skimage.measure
import xlsxwriter
import numpy as np

# initializing workbook
workbook = xlsxwriter.Workbook("ResultsQ1.xlsx")
worksheet = workbook.add_worksheet()
col = 0
row = 0
worksheet.write(row, col, 'Filter Name')
worksheet.write(row, col + 1, 'SSIM score value')
worksheet.write(row, col + 2, 'RMSE score value')
worksheet.write(row, col + 3, 'MSE score value')
row = row + 1


# function that writes results to the .xlsx file
def write_excel(filter_name, ssim_score, rmse_score, mse_score):
    global worksheet
    global row
    worksheet.write(row, col, filter_name)
    worksheet.write(row, col + 1, ssim_score)
    worksheet.write(row, col + 2, rmse_score)
    worksheet.write(row, col + 3, mse_score)
    row = row + 1


# choosing a random image from the folder \images_to_use
folder = r"C:\Users\green\PycharmProjects\ComputerVision\Project_1\images_to_use"
a = random.choice(os.listdir(folder))
img_RGB = cv2.imread('images_to_use/' + a)

# showing the image with no filter
cv2.namedWindow("No Filter", cv2.WINDOW_NORMAL)
cv2.imshow("No Filter", img_RGB)
cv2.resizeWindow("No Filter", 480, 360)

cv2.imwrite(r"C:\Users\green\PycharmProjects\ComputerVision\Project_1\output\no_filter.png", img_RGB)
# Averaging filter
blur = cv2.blur(img_RGB, (10, 10), 0)
cv2.namedWindow("Averaging Filter", cv2.WINDOW_NORMAL)
cv2.imshow("Averaging Filter", blur)
cv2.resizeWindow("Averaging Filter", 480, 360)
# calculate the similarity between the images
# compute the Structural Similarity Index (SSIM) between the two images
(ssim_score, _) = skimage.metrics.structural_similarity(img_RGB, blur, full=True, multichannel=True)
root_mse_score = skimage.metrics.normalized_root_mse(img_RGB, blur, normalization="euclidean")
mse_score = skimage.metrics.mean_squared_error(img_RGB, blur)
print("SSIM score: {:.4f}".format(ssim_score),
      ", Normalized root mse score: {:.4f}".format(root_mse_score),
      ", Mean squared error: {:.4f}".format(mse_score))
write_excel("Averaging Filter", ssim_score,  root_mse_score, mse_score)
cv2.waitKey()


# Gaussian Filter
blur = cv2.GaussianBlur(img_RGB, (11, 11), cv2.BORDER_DEFAULT)

cv2.namedWindow("Gauss blur Filter", cv2.WINDOW_NORMAL)  # this allows for resizing using mouse
cv2.imshow("Gauss blur Filter", blur)
cv2.resizeWindow("Gauss blur Filter", 480, 360)
cv2.resizeWindow("Gauss blur Filter", 480, 360)
# calculate the similarity between the images
# compute the Structural Similarity Index (SSIM) between the two images
(ssim_score, _) = skimage.metrics.structural_similarity(img_RGB, blur, full=True, multichannel=True)
root_mse_score = skimage.metrics.normalized_root_mse(img_RGB, blur, normalization="euclidean")
mse_score = skimage.metrics.mean_squared_error(img_RGB, blur)
print("SSIM score: {:.4f}".format(ssim_score),
      ", Normalized root mse score: {:.4f}".format(root_mse_score),
      ", Mean squared error: {:.4f}".format(mse_score))
write_excel("Guassian blur Filter", ssim_score, root_mse_score, mse_score)
cv2.waitKey()

# Median Filter
# This is highly effective in removing salt-and-pepper noise
blur = cv2.medianBlur(img_RGB, 9)

# illustrate the results
cv2.namedWindow("Median blur Filter", cv2.WINDOW_NORMAL)  # this allows for resizing using mouse
cv2.imshow("Median blur Filter", blur)
cv2.resizeWindow("Median blur Filter", 480, 360)
# calculate the similarity between the images
# compute the Structural Similarity Index (SSIM) between the two images
(ssim_score, _) = skimage.metrics.structural_similarity(img_RGB, blur, full=True, multichannel=True)
root_mse_score = skimage.metrics.normalized_root_mse(img_RGB, blur, normalization="euclidean")
mse_score = skimage.metrics.mean_squared_error(img_RGB, blur)
print("SSIM score: {:.4f}".format(ssim_score),
      ", Normalized root mse score: {:.4f}".format(root_mse_score),
      ", Mean squared error: {:.4f}".format(mse_score))
write_excel("Median blur Filter", ssim_score, root_mse_score, mse_score)
cv2.waitKey()

# Bilateral filter
blur = cv2.bilateralFilter(img_RGB, 20, 5, 5)

# illustrate the results
cv2.namedWindow("Bilateral Filter", cv2.WINDOW_NORMAL)  # this allows for resizing using mouse
cv2.imshow("Bilateral Filter", blur)
cv2.resizeWindow("Bilateral Filter", 480, 360)
# calculate the similarity between the images
# compute the Structural Similarity Index (SSIM) between the two images
(ssim_score, _) = skimage.metrics.structural_similarity(img_RGB, blur, full=True, multichannel=True)
root_mse_score = skimage.metrics.normalized_root_mse(img_RGB, blur, normalization="euclidean")
mse_score = skimage.metrics.mean_squared_error(img_RGB, blur)
print("SSIM score: {:.4f}".format(ssim_score),
      ", Normalized root mse score: {:.4f}".format(root_mse_score),
      ", Mean squared error: {:.4f}".format(mse_score))
write_excel("Bilateral blur Filter", ssim_score, root_mse_score, mse_score)
cv2.waitKey()

# destroy windows
cv2.destroyAllWindows()
workbook.close()
