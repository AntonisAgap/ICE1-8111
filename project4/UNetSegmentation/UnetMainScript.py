import keras
import matplotlib.pyplot as plt
import time
from keras.optimizers import *
from Project_4.UNetSegmentation.DataLoad import IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS,loadData
from Project_4.UNetSegmentation.UNET_Initialize import get_unet

# baseNumOfFilters = 16
# baseDropoutValue = 0.1
# batch_size = 16
# epochs = 10
inputImage = (IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS)
print(". Loading data....")
X_train, y_train, X_val, y_val, X_test, y_test = loadData()
print(". Data loaded successfully!")

print(". X_train shape: ", X_train.shape)
print(". X_val shape: ", X_val.shape)
print(". X_test shape: ", X_test.shape)


UnetCustMod = get_unet(n_filters=16,dropout=0.05,batchnorm=True)
print(". Compiling U-NET...")
UnetCustMod.compile(optimizer=Adam(lr=1e-5), loss="binary_crossentropy", metrics=["binary_accuracy"])
print(". U-NET compiled successfully!")

UnetCustMod.summary()

# train the model
callbacksOptions = [
    keras.callbacks.EarlyStopping(patience=10, verbose=1),
    keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    keras.callbacks.ModelCheckpoint('tmpSegCNN.h5', verbose=1, save_best_only=True, save_weights_only=False)
]

start_time_train = time.time()
history = UnetCustMod.fit(X_train, y_train, batch_size=32,
                          epochs=50, callbacks=callbacksOptions,
                          validation_data=(X_val, y_val))
end_time_train = time.time() - start_time_train

# save the final model, once trainng is completed
print(". Saving model...")
model_name = 'finalSegCNN.h5'
UnetCustMod.save(model_name)
print(". Model saved successfully!")

acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation binary accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure(figsize=(20, 10))
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()

plt.show()
