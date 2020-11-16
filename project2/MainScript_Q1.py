# /--------------------------------------/
# NAME: Antonis Agapiou
# E-MAIL: cs141081@uniwa.gr
# A.M: 711141081
# CLASS: Computer Vision
# LAB: CV Lab Group 1
# /--------------------------------------/

import cv2
import numpy as np
import random
import skimage
from matplotlib import pyplot as plt

# load the image to check
img_rgb = cv2.imread('image.png')
# load the template image we look for
template = cv2.imread('template.png')
template = skimage.util.img_as_float32(template, force_copy=False)


def TemplateMatching(img):
    # creating a clone of the image to not draw over the original one
    img_clone = img.copy()
    w, h, _ = template.shape
    # run the template matching
    res = cv2.matchTemplate(img_clone, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.60
    loc = np.where(res >= threshold)
    # mark the corresponding location(s)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_clone, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
    return img_clone



# # template matching with noise
noise = 0
for i in range(10): # 10
    if i == 5:
        noise = 0
    # detecting images using only noise
    if i < 5:
        noise_image = skimage.util.random_noise(img_rgb, mode='s&p', amount=noise)
        noise_image = skimage.util.img_as_float32(noise_image, force_copy=False)
        cv2.imshow("Detected,noise: " + str(noise), TemplateMatching(noise_image))
        cv2.waitKey(0)
        noise = round(noise + 0.05, 3)
    # detecting images using noise and a gaussian filter
    else:
        noise_image = skimage.util.random_noise(img_rgb, mode='s&p', amount=noise)
        blurred_image = cv2.GaussianBlur(noise_image, (5, 5), 0)
        blurred_image = skimage.util.img_as_float32(blurred_image, force_copy=False)
        cv2.imshow("Detected with Gaussian Blur,noise: " + str(noise), TemplateMatching(blurred_image))
        cv2.waitKey(0)
        noise = round(noise + 0.05, 3)
