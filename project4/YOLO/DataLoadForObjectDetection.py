from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from YOLO.SupportingFunctions.LoadingData import load_image_pixels
import sys
import requests
import os

input_w, input_h = 416, 416


# downloadImage()
path = 'C:/Users/green/Desktop/myDataset'

images = []
images_w = []
images_h = []
photo_filenames = []
for file in os.listdir(path):
    file_path = path + '/' + file
    image, image_w, image_h, = load_image_pixels(file_path, (input_w, input_h))
    images.append(image)
    images_w.append(image_w)
    images_h.append(image_h)
    photo_filenames.append(file_path)
