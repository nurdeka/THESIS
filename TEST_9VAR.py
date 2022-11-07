# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 08:04:28 2022

@author: nurde
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

#import autokeras as ak

import os
import zipfile

from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import keras,os
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense, Conv3D, MaxPool3D , Flatten, MaxPooling3D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import timeit
from numpy import load
import pickle
roi = 25
channel =11

dict_data = load(f'dataset_{roi}x{roi}.npz')
dict_data.files
X_trains = dict_data["X_train"]
X_tests = dict_data["X_test"]
y_train = dict_data["y_train"]
y_test = dict_data["y_test"]
print(X_trains.shape)
print(X_tests.shape)
print(y_train.shape)
print(y_test.shape)

# Eksperimen Tanpa SLope TPI
X_train = np.stack((X_trains[:,:,:,0],  X_trains[:,:,:,3], X_trains[:,:,:,4],
                     X_trains[:,:,:,5], X_trains[:,:,:,6],  X_trains[:,:,:,7], X_trains[:,:,:,8], X_trains[:,:,:,9], X_trains[:,:,:,10]), axis=-1)
X_test = np.stack((X_tests[:,:,:,0],  X_tests[:,:,:,3], X_tests[:,:,:,4],
                     X_tests[:,:,:,5], X_tests[:,:,:,6], X_tests[:,:,:,7], X_tests[:,:,:,8], X_tests[:,:,:,9], X_tests[:,:,:,10]), axis=-1)

print(X_train.shape)
print(X_test.shape)

#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print('Dimensi X_train : ', X_train.shape)
print('Dimensi X_val : ',X_val.shape)
print('Dimensi y_train : ',y_train.shape)
print('Dimensi y_val : ',y_val.shape)
# %%
window = X_train.shape[1]
channel = X_train.shape[3]
print(window)
print(channel)


#%%
#MODEL_1
def get_model(width=window, height=window, depth=channel):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    #x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(width=window, height=window, depth=channel)
model.summary()

#%%
# Compile the model
learning_rate =0.001 
batch_size = 3
verbosity = 0
model.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),metrics=["accuracy"])
# Fit the model
history = model.fit(X_train, y_train, epochs=200, validation_data=[X_val, y_val], callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=20)])

#%%
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Evaluate the test performance of the tuned model Hyperband
eval_result = model.evaluate(X_test, y_test)
print("[test loss, test accuracy]:", eval_result)

from sklearn.metrics import roc_curve
y_pred_keras = model.predict(X_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)

from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

#%%
#directory = "H:/My Drive/DATA/MODEL/"
eval = int(eval_result[1]*1000)
auc = int(auc_keras*1000)
name = (f'model9var_{roi}_{eval}_{auc}.h5')
os.chdir("/content/gdrive/MyDrive/DATA/MODEL" )
path = os.path.join(os.getcwd(), name)
model.save(path)
print('Saved trained model at %s ' % path)

from numpy import savez_compressed
# save to npy file
name_1 = (f'model9var_{roi}_{eval}_{auc}_history.npz')
path = os.path.join(os.getcwd(), name_1)
savez_compressed(path)
print('Saved trained history at %s ' % path)