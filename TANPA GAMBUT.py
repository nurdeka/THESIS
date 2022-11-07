# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:36:53 2022

@author: nurde
"""

import pandas as pd
import numpy as np
from numpy import load
import os
import numpy as np
import tensorflow as tf
import os
import zipfile

from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import keras,os
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense, Conv3D, MaxPool3D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import timeit
from numpy import savez_compressed
from numpy import load

os.chdir('C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616')
os.getcwd()

roi = 5

dict_data = load(f'dataset_{roi}x{roi}.npz')
dict_data.files
X_train = dict_data["X_train"]
X_test = dict_data["X_test"]
y_train = dict_data["y_train"]
y_test = dict_data["y_test"]
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val= train_test_split(X_trains, y_train, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_val.shape)


import joblib
from tensorflow.keras.models import load_model

filename1 = 'finalized_model_RF.sav'
filename2 = 'finalized_model_GBC.sav'
filename3 = 'finalized_model_SVC.sav'
filename4 = 'model_acc_914_auc_974.h5'
filename5 = 'Finalized_nn.h5'
rf_model = joblib.load(filename1)
gb_model = joblib.load(filename2)
sv_model = joblib.load(filename3)
cn_model = load_model(filename4)
nn_model = load_model(filename5)

# Eksperimen Tanpa Gambut
X_trains = np.stack((X_train[:,:,:,0], X_train[:,:,:,1], X_train[:,:,:,2], X_train[:,:,:,3], X_train[:,:,:,4],
                     X_train[:,:,:,5], X_train[:,:,:,6], X_train[:,:,:,8], X_train[:,:,:,9], X_train[:,:,:,10]), axis=-1)
X_tests = np.stack((X_test[:,:,:,0], X_test[:,:,:,1], X_test[:,:,:,2], X_test[:,:,:,3], X_test[:,:,:,4],
                     X_test[:,:,:,5], X_test[:,:,:,6], X_test[:,:,:,8], X_test[:,:,:,9], X_test[:,:,:,10]), axis=-1)
print(X_trains.shape)
print(X_tests.shape)

window = X_train.shape[1]
channel = X_train.shape[3]
print(window)
print(channel)
#%% CNN 5x5
def get_model(width=window, height=window, depth=channel):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    #x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    #x = layers.MaxPool3D(pool_size=2)(x)
    #x = layers.BatchNormalization()(x)

    #x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    #x = layers.MaxPool3D(pool_size=2)(x)
    #x = layers.BatchNormalization()(x)

    #x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    #x = layers.MaxPool3D(pool_size=2)(x)
    #x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=112, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
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
# Compile model.
initial_learning_rate = 0.0001
#lr_schedule = keras.optimizers.schedules.ExponentialDecay(
 #   initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
#)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate),
    metrics=["acc"],
)


# Train the model, doing validation at the end of each epoch
epochs = 200
model.fit(
    X_train,y_train,
    validation_data=[X_val, y_val],
    epochs=epochs,
    #shuffle=True,
    verbose=2,
    callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=20)],
)

# %%
# evaluate the model
scores = model.evaluate(X_tests, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#%%
# Evaluate the test performance of the tuned model Hyperband
eval_result = model.evaluate(X_tests, y_test)
print("[test loss, test accuracy]:", eval_result)

from sklearn.metrics import roc_curve
y_pred_keras = model.predict(X_tests).ravel()
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

# %%
model.save("model_nopeat_cnn.h5")
print("Saved model to disk")