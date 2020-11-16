# /--------------------------------------/
# NAME: Antonis Agapiou
# E-MAIL: cs141081@uniwa.gr
# A.M: 711141081
# CLASS: Computer Vision
# LAB: CV Lab Group 1
# /--------------------------------------/

import cv2
from matplotlib import pyplot as plt
import math

# load the image to check
img_rgb = cv2.imread('image.png')
# load the template image we look for
template = cv2.imread('template.png')

# deleting the last row and column of image to have odd number of rows and columns
template = template[0:template.shape[0] - 1, 0:template.shape[1] - 1]

# getting images' dimensions
w, h, _ = template.shape
W, H, _ = img_rgb.shape

# choosing bin size using Sturge's rule
bins = 1 + 3.322 * math.log(w * h)
bins = int(round(bins, 1))

# plot color histogram

# for i, col in enumerate(['b', 'g', 'r']):
#     hist = cv2.calcHist([template], [i], None, [256], [0, 256])
#     plt.plot(hist, color=col)
#     plt.xlim([0, 256])
#
# plt.show()

# calculating RGB histogram for template image
template_hist = cv2.calcHist([template], [0, 1, 2], None, [bins, bins, bins],
                             [0, 256, 0, 256, 0, 256])

template_hist = cv2.normalize(template_hist, template_hist).flatten()
# template_hist = cv2.calcHist(template, [0], None, [256], [0, 256])

color = (0, 255, 0)
thickness = 1

detected_image_comp = img_rgb.copy()

for i in range(0, W, 5):
    for j in range(0, H, 5):
        image = img_rgb[i:i + w, j:j + h]
        # image = img_rgb[i-w//2:i+w//2, j-h//2:j+w//2]
        if image.shape == template.shape:
            # calculating image's patch RGB histogram
            hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins],
                                [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            # comparing histograms using CV_COMP_CORREL
            result = cv2.compareHist(template_hist, hist, 0)
            start_point = (i, j)
            end_point = (i + w, j + h)
            if result > 0.60:
                # drawing rectangle around image patch with <60% correlation
                cv2.rectangle(detected_image_comp, (i, j), (i + w, j + h), color, thickness)

cv2.imshow("comp_correl_detected", detected_image_comp)
# cv2.imwrite("comp_correl_detected.png", detected_image_comp)
cv2.waitKey()


# doing the same thing with different metric and threshold

detected_image_comp2 = img_rgb.copy()
for i in range(0, W, 5):
    for j in range(0, H, 5):
        image2 = img_rgb[i:i + w, j:j + h]
        # image = img_rgb[i-w//2:i+w//2, j-h//2:j+w//2]
        if image2.shape == template.shape:
            # calculating image's patch RGB histogram
            hist = cv2.calcHist([image2], [0, 1, 2], None, [bins, bins, bins],
                                [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            # comparing histograms using CV_COMP_BHATTACHARYYA
            result = cv2.compareHist(template_hist, hist, 3)
            if result < 0.40:
                # drawing rectangle around image patch with <60% correlation
                cv2.rectangle(detected_image_comp2, (i, j), (i + w, j + h), color, thickness)

cv2.imshow("comp_bhatta_detected", detected_image_comp2)
# cv2.imwrite("comp_bhatta_detected.png", detected_image_comp2 )
cv2.waitKey()