import random
import os
import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

# from Project_4.UNetSegmentation.UnetMainScript import X_test, y_test, end_time_train

from Project_4.UNetSegmentation.DataLoad import IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS,loadData
X_train, y_train, X_val, y_val, X_test, y_test = loadData()

print(". Loading model...")
tmpCustMod = keras.models.load_model("finalSegCNN.h5")
print(". Model loaded successfully!")

# calculate some common performance scores
score = tmpCustMod.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

start_time_test = time.time()
print(". Segmentation on", X_test.shape[0], "samples...")
y_pred = tmpCustMod.predict(X_test, batch_size=32)
print(". Segmentation done!")
end_time_test = time.time() - start_time_test

# count = 0
# print(". Saving results at" + path_temp_output_images + "...")
# for image in y_pred:
#     cv2.imwrite(os.path.join(path_temp_output_images, str(count + 1) + ".png"), image)
#     count = count + 1
# print(". Results saved successfully!")

# print(". Training time: " + str(end_time_train) + " seconds.")
print(". Testing time: " + str(end_time_test) + " seconds on", X_test.shape[0], "images.")

print(". Plotting random image...")
i = random.choice(range(len(X_test)))
fig, axs = plt.subplots(1, 3, figsize=(16, 16))
axs[0].set_title("X-Ray")
axs[0].imshow(np.squeeze(X_test[i], axis=2), cmap="gray")
axs[1].set_title("Ground Truth")
axs[1].imshow(np.squeeze(y_test[i], axis=2), cmap="gray")
axs[2].set_title("Predicted")
axs[2].imshow(np.squeeze(y_pred[i], axis=2), cmap="gray")
plt.show()
