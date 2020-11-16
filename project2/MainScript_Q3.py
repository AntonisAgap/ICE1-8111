# /--------------------------------------/
# NAME: Antonis Agapiou
# E-MAIL: cs141081@uniwa.gr
# A.M: 711141081
# CLASS: Computer Vision
# LAB: CV Lab Group 1
# /--------------------------------------/

import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans, OPTICS, AgglomerativeClustering
from skimage.color import label2rgb
from sklearn.feature_extraction.image import grid_to_graph
import skimage

# load the image to check
original_image = cv2.imread("fire.jpg")
a_image = cv2.imread("annotated_img.jpg", 0)

# display the original image
cv2.imshow("Original image", original_image)
cv2.waitKey()
cv2.destroyAllWindows()

original_image_shape = original_image.shape


def CalculatingMetrics(annotated_image, segmented_image):
    height, width = annotated_image.shape

    annotation_of_image = []
    for i in range(height):
        for j in range(width):
            if annotated_image[i, j] == 255:
                annotation_of_image.append([i, j])

    clusters_pixels_value = []
    # getting the counterpart of the annotated image on the segmented image
    for pixel in annotation_of_image:
        cluster_pixel_value = segmented_image[pixel[0], pixel[1]]
        clusters_pixels_value.append(cluster_pixel_value)

    # getting the most frequent value of RGB pixel
    values, counts = np.unique(clusters_pixels_value, return_counts=True, axis=0)
    main_cluster_RGB_value = values[counts == np.max(counts),]
    num_pixels = np.max(counts)

    # true positives are the pixels that belong to the annotation and to the cluster
    true_positives = num_pixels
    # false negative are the pixels that belong to the annotation but not the cluster
    false_negatives = len(clusters_pixels_value) - true_positives

    count = 0
    for i in range(height):
        for j in range(width):
            if np.array_equal(segmented_image[i, j], main_cluster_RGB_value[0]):
                count = count + 1
    # false positives are the pixels that belong to the cluster but not the annotation
    false_positives = count - true_positives

    precision = round(true_positives / (true_positives + false_positives), 2)
    print(" Precision of model is: ", precision)
    f1_score = round(2 * true_positives / ((2 * true_positives) + false_positives + false_negatives), 2)
    print(" F1_score of model is: ", f1_score)


def Clustering(image, amount_noise):
    # transforming image to appropriate input for cluster
    flatImg = np.reshape(image, [-1, 3])
    # Using Meanshift algorithm
    # Estimating bandwidth for meanshift algorithm
    bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    print(". Using MeanShift Algorithm with", amount_noise, "noise.")
    ms.fit(flatImg)
    labels = ms.labels_
    # Finding and displaying the number of clusters
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print(". Number of estimated clusters using MeanShift: %d" % n_clusters_)
    # Displaying segmented image using MeanShift
    ms_segmentedImg = np.reshape(labels, original_image_shape[:2])
    ms_segmentedImg = label2rgb(ms_segmentedImg) * 255

    cv2.imshow("MeanShift segments", ms_segmentedImg.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("MeanShiftSegmentedImage.png", ms_segmentedImg)
    print(". Done!")
    print(". Calculating scores")
    CalculatingMetrics(a_image, ms_segmentedImg)

    # Agglomerative clustering algorithm
    x, y, z = original_image.shape
    connectivity = grid_to_graph(n_x=x, n_y=y)
    print(". Using Agglomerative Clustering Algorithm with", amount_noise, "noise.")
    ac = AgglomerativeClustering(n_clusters=n_clusters_, linkage="ward", connectivity=connectivity)
    ac.fit(flatImg)
    labels = ac.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print(" Number of estimated clusters using Agglomerative clustering: %d" % n_clusters_)

    # Displaying segmented image using KMeans
    ac_segmentedImg = np.reshape(labels, original_image_shape[:2])
    ac_segmentedImg = label2rgb(ac_segmentedImg) * 255
    cv2.imshow("Agglomerative clustering segmented image", ac_segmentedImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("AgglomerativeSegmentedImage.png", ac_segmentedImg)
    print(". Done!")
    print(". Calculating scores")
    CalculatingMetrics(a_image, ac_segmentedImg)

    # KMeans algorithm
    print(". Using KMeans Clustering Algorithm with", amount_noise, "noise.")
    km = KMeans()
    km.fit(flatImg)
    labels = km.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print(" Number of estimated clusters using KMeans: %d" % n_clusters_)

    # Displaying segmented image using KMeans
    km_segmentedImg = np.reshape(labels, original_image_shape[:2])
    km_segmentedImg = label2rgb(km_segmentedImg) * 255
    cv2.imshow("KMeans segmented image", km_segmentedImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("KMeansSegmentedImage.png", km_segmentedImg)
    print(". Done!")
    print(". Calculating scores")
    CalculatingMetrics(a_image, km_segmentedImg)


noise = 0
for i in range(5):
    noise_image = skimage.util.random_noise(original_image, mode='s&p', amount=noise)
    noise_image = skimage.util.img_as_float32(noise_image, force_copy=False)
    Clustering(noise_image, noise)
    noise = noise + 0.05
