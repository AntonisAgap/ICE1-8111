import keras
# Import stuff from DataLoad
from Project_4.MonkeySpeciesCNN.DataLoad import X_train, y_train, X_test, y_test, X_val, y_val, inputImageSize, \
    init_excel, write_excel
import matplotlib.pyplot as plt  # plotting library
import numpy as np
from keras import regularizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import xlsxwriter

workbook, worksheet, col, row = init_excel()

epochs = 15
batch_size = 32

IMG_HEIGHT = inputImageSize[0]
IMG_WIDTH = inputImageSize[1]
IMG_CHANNELS = inputImageSize[2]

print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'validation samples')
print(X_test.shape[0], 'test samples')

y_test_encoded = y_test
y_train_encoded = y_train

# convert class vectors to binary class matrices
num_classes = np.unique(y_train).__len__()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

inputs = keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = keras.layers.Lambda(lambda x: x / 255)(inputs)  # normalize the input
conv1 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_regularizer=regularizers.l2(0.01))(s)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_regularizer=regularizers.l2(0.01))(pool1)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = keras.layers.Conv2D(filters=32 * 2, kernel_size=(3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(0.01))(pool2)
conv4 = keras.layers.Conv2D(filters=32 * 2, kernel_size=(3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(0.01))(conv3)
pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
drop3 = keras.layers.Dropout(0.25)(pool3)
flat1 = keras.layers.Flatten()(drop3)
dense1 = keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(flat1)
drop4 = keras.layers.Dropout(0.25)(dense1)
outputs = keras.layers.Dense(num_classes, activation='softmax')(dense1)

CNNmodel = keras.Model(inputs=[inputs], outputs=[outputs])

sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# RMSprop = keras.optimizers.RMSprop(0.001)
print(". Compiling CNN model...")
CNNmodel.compile(optimizer=sgd, loss=keras.losses.categorical_crossentropy, metrics=['acc'])
print(". Model compiled successfully!")
# # print model summary
CNNmodel.summary()

# fit model parameters, given a set of training data
callbacksOptions = [
    # keras.callbacks.EarlyStopping(patience=15, verbose=1),
    # keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0001, verbose=1),
    keras.callbacks.ModelCheckpoint('tmpCNN.h5', verbose=1, save_best_only=True, save_weights_only=False)]

train_start = time.time()
history = CNNmodel.fit(X_train, y_train, batch_size=batch_size, shuffle=True, epochs=epochs, verbose=1,
                       callbacks=callbacksOptions, validation_data=(X_val, y_val))
train_end = time.time() - train_start

# calculate some common performance scores
# score = CNNmodel.evaluate(X_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure(figsize=(20, 10))
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()

plt.show()

loaded_model = keras.models.load_model('tmpCNN.h5')
print("Model Loaded Successfully")

test_start = time.time()
y_test_predictions_vectorized = loaded_model.predict(X_test)
test_end = time.time() - test_start
y_test_predictions = np.argmax(y_test_predictions_vectorized, axis=1)

y_train_predictions_vectorized = loaded_model.predict(X_train)
y_train_predictions = np.argmax(y_train_predictions_vectorized, axis=1)

acc_train = accuracy_score(y_train_encoded, y_train_predictions)
pre_train = precision_score(y_train_encoded, y_train_predictions, average='macro')
rec_train = recall_score(y_train_encoded, y_train_predictions, average='macro')
f1_train = f1_score(y_train_encoded, y_train_predictions, average='macro')

acc_test = accuracy_score(y_test_encoded, y_test_predictions)
pre_test = precision_score(y_test_encoded, y_test_predictions, average='macro')
rec_test = recall_score(y_test_encoded, y_test_predictions, average='macro')
f1_test = f1_score(y_test_encoded, y_test_predictions, average='macro')

print('Accuracy of train score is: {:.2f}.'.format(acc_train))
print('Precision of train score is: {:.2f}.'.format(pre_train))
print('Recall score of train is: {:.2f}.'.format(rec_train))
print('F1 score of train is: {:.2f}.'.format(f1_train))

print('Accuracy of test score is: {:.2f}.'.format(acc_test))
print('Precision of test score is: {:.2f}.'.format(pre_test))
print('Recall score of test is: {:.2f}.'.format(rec_test))
print('F1 score of test is: {:.2f}.'.format(f1_test))
print('Training time: ' + str(train_end) + ' seconds')
print('Testing time: ' + str(test_end) + ' seconds on', X_test.shape[0], ' images')

write_excel("CNN", "60/40", acc_train, pre_train, rec_train,
            f1_train, acc_test, pre_test, rec_test, f1_test, worksheet, row, col)

workbook.close()
