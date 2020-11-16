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
import numpy as np
import time
import skimage.metrics
import skimage.measure
import xlsxwriter

# initializing workbook
workbook = xlsxwriter.Workbook("ResultsQ2.xlsx")
worksheet = workbook.add_worksheet()
col = 0
row = 0
worksheet.write(row, col, 'Image ID')
worksheet.write(row, col + 1, 'Noise Type')
worksheet.write(row, col + 2, 'Filter Name')
worksheet.write(row, col + 3, 'SSIM score value')
worksheet.write(row, col + 4, 'PSNR score value')
row = row + 1


# function that writes results to the .xlsx file
def write_excel(image_id, noise_type, filter_name, ssim_score, psnr_score):
    global worksheet
    global row
    worksheet.write(row, col, image_id)
    worksheet.write(row, col + 1, noise_type)
    worksheet.write(row, col + 2, filter_name)
    worksheet.write(row, col + 3, ssim_score)
    worksheet.write(row, col + 4, psnr_score)
    row = row + 1


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = skimage.util.img_as_float32(img)
            images.append(img)
    return images


# Applying noise to the images and saving generated images to lists

noise_images_pdn = []
noise_images_gn = []
noise_images_sp = []
images_folder = r"C:\Users\green\PycharmProjects\ComputerVision\Project_1\images_to_use"
original_images = load_images_from_folder(images_folder)
for original_image in original_images:
    noise_image_sp = skimage.util.random_noise(original_image, mode='s&p', amount=0.1)
    nosie_image_sp = skimage.util.img_as_float32(noise_image_sp)
    noise_images_sp.append(noise_image_sp)
    noise_image_gn = skimage.util.random_noise(original_image, mode='gaussian', mean=0, var=1)
    noise_image_gn = skimage.util.img_as_float32(noise_image_gn)
    noise_images_gn.append(noise_image_gn)
    noise_image_pdn = skimage.util.random_noise(original_image, mode='poisson')
    noise_image_pdn = skimage.util.img_as_float32(noise_image_pdn)
    noise_images_pdn.append(noise_image_pdn)


for i in range(len(original_images)):
    cv2.namedWindow("Original Image No." + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image No." + str(i), original_images[i])
    cv2.resizeWindow("Original Image No." + str(i), 480, 360)
    cv2.waitKey()

    cv2.namedWindow("Original (sp noise) Image No." + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Original (sp noise) Image No." + str(i), noise_images_sp[i])
    cv2.resizeWindow("Original (sp noise) Image No." + str(i), 480, 360)
    cv2.waitKey()
    cv2.namedWindow("Original (gaussian noise) Image No." + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Original (gaussian noise) Image No." + str(i), noise_images_gn[i])
    cv2.resizeWindow("Original (gaussian noise) Image No." + str(i), 480, 360)
    cv2.waitKey()
    cv2.namedWindow("Original (poisson noise) Image No." + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Original (poisson noise) Image No." + str(i), noise_images_pdn[i])
    cv2.resizeWindow("Original (poisson noise) Image No." + str(i), 480, 360)
    cv2.waitKey()

cv2.destroyAllWindows()

for i in range(len(original_images)):
    avg_blur_sp = cv2.blur(noise_images_sp[i], (5, 5), 0)
    gaussian_blur_sp = cv2.GaussianBlur(noise_images_sp[i], (5, 5), 0)
    bilateral_blur_sp = cv2.bilateralFilter(noise_images_sp[i], 9, 5, 5)
    median_blur_sp = cv2.medianBlur(noise_images_sp[i], 5, 0)

    cv2.namedWindow("Salt & pepper noise with no Blur Image No." + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Salt & pepper noise with no Blur Image No." + str(i), noise_images_sp[i])
    cv2.resizeWindow("Salt & pepper noise with no Blur Image No." + str(i), 480, 360)
    cv2.waitKey()

    cv2.namedWindow("Salt & pepper noise with Averaging Blur Image No." + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Salt & pepper noise with Averaging Blur Image No." + str(i), avg_blur_sp)
    cv2.resizeWindow("Salt & pepper noise with Averaging Blur Image No." + str(i), 480, 360)
    cv2.waitKey()

    (ssim_score, _) = skimage.metrics.structural_similarity(original_images[i], avg_blur_sp, full=True,
                                                            multichannel=True)
    psnr_score = skimage.metrics.peak_signal_noise_ratio(original_images[i], avg_blur_sp)
    print("SSIM score (Salt & pepper noise) with Averaging Blur:  ", ssim_score)
    print("PSNR score (Salt & pepper noise) with Averaging Blur: ", psnr_score)
    write_excel(i, "Salt & pepper noise", "Averaging Blur", ssim_score, psnr_score)

    cv2.namedWindow("Salt & pepper noise with Gaussian Blur Image No." + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Salt & pepper noise with Gaussian Blur Image No." + str(i), gaussian_blur_sp)
    cv2.resizeWindow("Salt & pepper noise with Gaussian Blur Image No." + str(i), 480, 360)
    cv2.waitKey()

    (ssim_score, _) = skimage.metrics.structural_similarity(original_images[i], gaussian_blur_sp, full=True,
                                                            multichannel=True)
    psnr_score = skimage.metrics.peak_signal_noise_ratio(original_images[i], gaussian_blur_sp)
    write_excel(i, "Salt & pepper noise", "Gaussian Blur", ssim_score, psnr_score)

    print("SSIM score (Salt & pepper noise) with Gaussian Blur:  ", ssim_score)
    print("PSNR score (Salt & pepper noise) with Gaussian Blur: ", psnr_score)

    cv2.namedWindow("Salt & pepper noise with Bilateral Blur Image No." + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Salt & pepper noise with Bilateral Blur Image No." + str(i), bilateral_blur_sp)
    cv2.resizeWindow("Salt & pepper noise with Bilateral Blur Image No." + str(i), 480, 360)
    cv2.waitKey()

    (ssim_score, _) = skimage.metrics.structural_similarity(original_images[i], bilateral_blur_sp, full=True,
                                                            multichannel=True)
    psnr_score = skimage.metrics.peak_signal_noise_ratio(original_images[i], bilateral_blur_sp)
    write_excel(i, "Salt & pepper noise", "Bilateral Blur", ssim_score, psnr_score)

    print("SSIM score (Salt & pepper noise) with Bilateral Blur:  ", ssim_score)
    print("PSNR score (Salt & pepper noise) with Bilateral Blur: ", psnr_score)

    cv2.namedWindow("Salt & pepper noise with Median Blur Image No." + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Salt & pepper noise with Median Blur Image No." + str(i), median_blur_sp)
    cv2.resizeWindow("Salt & pepper noise with Median Blur Image No." + str(i), 480, 360)
    cv2.waitKey()

    (ssim_score, _) = skimage.metrics.structural_similarity(original_images[i], median_blur_sp, full=True,
                                                            multichannel=True)
    psnr_score = skimage.metrics.peak_signal_noise_ratio(original_images[i], median_blur_sp)
    write_excel(i, "Salt & pepper noise", "Median Blur", ssim_score, psnr_score)

    print("SSIM score (Salt & pepper noise) with Median Blur:  ", ssim_score)
    print("PSNR score (Salt & pepper noise) with Median Blur: ", psnr_score)

    avg_blur_gn = cv2.blur(noise_images_gn[i], (5, 5), 0)
    gaussian_blur_gn = cv2.GaussianBlur(noise_images_gn[i], (5, 5), 0)
    bilateral_blur_gn = cv2.bilateralFilter(noise_images_gn[i], 9, 5, 5)
    median_blur_gn = cv2.medianBlur(noise_images_gn[i], 5, 0)

    cv2.namedWindow("Gaussian noise with no Blur Image No." + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Gaussian noise with no Blur Image No." + str(i), noise_images_gn[i])
    cv2.resizeWindow("Gaussian noise with no Blur Image No." + str(i), 480, 360)
    cv2.waitKey()

    cv2.namedWindow("Gaussian noise with Averaging Blur Image No." + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Gaussian noise with Averaging Blur Image No." + str(i), avg_blur_gn)
    cv2.resizeWindow("Gaussian noise with Averaging Blur Image No." + str(i), 480, 360)
    cv2.waitKey()

    (ssim_score, _) = skimage.metrics.structural_similarity(original_images[i], avg_blur_gn, full=True,
                                                            multichannel=True)
    psnr_score = skimage.metrics.peak_signal_noise_ratio(original_images[i], avg_blur_gn)
    print("SSIM score (Gaussian noise) with Averaging Blur:  ", ssim_score)
    print("PSNR score (Gaussian noise) with Averaging Blur: ", psnr_score)
    write_excel(i, "Gaussian noise", "Averaging Blur", ssim_score, psnr_score)

    cv2.namedWindow("Gaussian noise with Gaussian Blur Image No." + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Gaussian noise with Gaussian Blur Image No." + str(i), gaussian_blur_gn)
    cv2.resizeWindow("Gaussian noise with Gaussian Blur Image No." + str(i), 480, 360)
    cv2.waitKey()

    (ssim_score, _) = skimage.metrics.structural_similarity(original_images[i], gaussian_blur_gn, full=True,
                                                            multichannel=True)
    psnr_score = skimage.metrics.peak_signal_noise_ratio(original_images[i], gaussian_blur_gn)
    write_excel(i, "Gaussian noise", "Gaussian Blur", ssim_score, psnr_score)

    print("SSIM score (Gaussian noise) with Gaussian Blur:  ", ssim_score)
    print("PSNR score (Gaussian noise) with Gaussian Blur: ", psnr_score)

    cv2.namedWindow("Gaussian noise with Bilateral Blur Image No." + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Gaussian noise with Bilateral Blur Image No." + str(i), bilateral_blur_gn)
    cv2.resizeWindow("Gaussian noise with Bilateral Blur Image No." + str(i), 480, 360)
    cv2.waitKey()

    (ssim_score, _) = skimage.metrics.structural_similarity(original_images[i], bilateral_blur_gn, full=True,
                                                            multichannel=True)
    psnr_score = skimage.metrics.peak_signal_noise_ratio(original_images[i], bilateral_blur_gn)
    write_excel(i, "Gaussian noise", "Bilateral Blur", ssim_score, psnr_score)

    print("SSIM score (Gaussian noise) with Bilateral Blur:  ", ssim_score)
    print("PSNR score (Gaussian noise) with Bilateral Blur: ", psnr_score)

    cv2.namedWindow("Gaussian noise with Median Blur Image No." + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Gaussian noise with Median Blur Image No." + str(i), median_blur_gn)
    cv2.resizeWindow("Gaussian noise with Median Blur Image No." + str(i), 480, 360)
    cv2.waitKey()

    (ssim_score, _) = skimage.metrics.structural_similarity(original_images[i], median_blur_gn, full=True,
                                                            multichannel=True)
    psnr_score = skimage.metrics.peak_signal_noise_ratio(original_images[i], median_blur_gn)
    print("SSIM score (Gaussian noise) with Median Blur:  ", ssim_score)
    print("PSNR score (Gaussian noise) with Median Blur: ", psnr_score)
    write_excel(i, "Gaussian noise", "Median Blur", ssim_score, psnr_score)

    avg_blur_pdn = cv2.blur(noise_images_pdn[i], (5, 5), 0)
    gaussian_blur_pdn = cv2.GaussianBlur(noise_images_pdn[i], (5, 5), 0)
    bilateral_blur_pdn = cv2.bilateralFilter(noise_images_pdn[i], 9, 5, 5)
    median_blur_pdn = cv2.medianBlur(noise_images_pdn[i], 5, 0)

    cv2.namedWindow("Poisson noise with no Blur Image No." + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Poisson noise with no Blur Image No." + str(i), noise_images_pdn[i])
    cv2.resizeWindow("Poisson noise with no Blur Image No." + str(i), 480, 360)
    cv2.waitKey()

    cv2.namedWindow("Poisson noise with Averaging Blur Image No." + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Poisson noise with Averaging Blur Image No." + str(i), avg_blur_pdn)
    cv2.resizeWindow("Poisson noise with Averaging Blur Image No." + str(i), 480, 360)
    cv2.waitKey()

    (ssim_score, _) = skimage.metrics.structural_similarity(original_images[i], avg_blur_pdn, full=True,
                                                            multichannel=True)
    psnr_score = skimage.metrics.peak_signal_noise_ratio(original_images[i], avg_blur_pdn)
    write_excel(i, "Poisson noise", "Averaging Blur", ssim_score, psnr_score)

    print("SSIM score (Poisson noise) with Averaging Blur:  ", ssim_score)
    print("PSNR score (Poisson noise) with Averaging Blur: ", psnr_score)

    cv2.namedWindow("Poisson noise with Gaussian Blur Image No." + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Poisson noise with Gaussian Blur Image No." + str(i), gaussian_blur_pdn)
    cv2.resizeWindow("Poisson noise with Gaussian Blur Image No." + str(i), 480, 360)
    cv2.waitKey()

    (ssim_score, _) = skimage.metrics.structural_similarity(original_images[i], gaussian_blur_pdn, full=True,
                                                            multichannel=True)
    psnr_score = skimage.metrics.peak_signal_noise_ratio(original_images[i], gaussian_blur_pdn)
    write_excel(i, "Poisson noise", "Gaussian Blur", ssim_score, psnr_score)

    print("SSIM score (Poisson noise) with Gaussian Blur:  ", ssim_score)
    print("PSNR score (Poisson noise) with Gaussian Blur: ", psnr_score)

    cv2.namedWindow("Poisson noise with Bilateral Blur Image No." + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Poisson noise with Bilateral Blur Image No." + str(i), bilateral_blur_pdn)
    cv2.resizeWindow("Poisson noise with Bilateral Blur Image No." + str(i), 480, 360)
    cv2.waitKey()

    (ssim_score, _) = skimage.metrics.structural_similarity(original_images[i], bilateral_blur_pdn, full=True,
                                                            multichannel=True)
    psnr_score = skimage.metrics.peak_signal_noise_ratio(original_images[i], bilateral_blur_pdn)
    write_excel(i, "Poisson noise", "Bilateral Blur", ssim_score, psnr_score)

    print("SSIM score (Poisson noise) with Bilateral Blur:  ", ssim_score)
    print("PSNR score (Poisson noise) with Bilateral Blur: ", psnr_score)

    cv2.namedWindow("Poisson noise with Median Blur Image No." + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Poisson noise with Median Blur Image No." + str(i), median_blur_pdn)
    cv2.resizeWindow("Poisson noise with Median Blur Image No." + str(i), 480, 360)
    cv2.waitKey()

    (ssim_score, _) = skimage.metrics.structural_similarity(original_images[i], median_blur_pdn, full=True,
                                                            multichannel=True)
    psnr_score = skimage.metrics.peak_signal_noise_ratio(original_images[i], median_blur_pdn)
    write_excel(i, "Poisson noise", "Median Blur", ssim_score, psnr_score)

    print("SSIM score (Poisson noise) with Median Blur:  ", ssim_score)
    print("PSNR score (Poisson noise) with Median Blur: ", psnr_score)
    cv2.destroyAllWindows()

workbook.close()
