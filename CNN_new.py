# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:54:24 2022

@author: nurde
"""
#%%
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

#%%
os.chdir('C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616')
os.getcwd()

#%%
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

#%%
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_val.shape)
# %%
window = X_train.shape[1]
channel = X_train.shape[3]
print(window)
print(channel)



#%% CNN 5x5
#MODEL_1
def get_model(width=window, height=window, depth=channel):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=112, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(units=64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model_1 = keras.Model(inputs, outputs, name="3dcnn")
    return model_1


# Build model.
model_1 = get_model(width=window, height=window, depth=channel)
model_1.summary()


 #%%
 #%%
 # Compile model
model_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model_1.fit(X_train, y_train, validation_split=0.2, shuffle=True,
                      epochs=200, batch_size=10, verbose=0,
                      callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=20)])
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# %%
# evaluate the model
scores = model_1.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model_1.metrics_names[1], scores[1]*100))

#%%
# Evaluate the test performance of the tuned model Hyperband
eval_result = model_1.evaluate(X_test, y_test)
print("[test loss, test accuracy]:", eval_result)

from sklearn.metrics import roc_curve
y_pred_keras = model_1.predict(X_test).ravel()
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
model_1.save("model_keras_acc_909_auc_968.h5")
print("Saved model to disk")
#%% CNN DARI KERAS TUNER VAL ACC 94%
#MODEL_2
def get_model(width=window, height=window, depth=channel):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(units=32, activation="relu")(x)
    x = layers.Dropout(0.1)(x)



    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model_2 = keras.Model(inputs, outputs, name="3dcnn")
    return model_2


# Build model.
model_2 = get_model(width=window, height=window, depth=channel)
model_2.summary()

#%%
# Compile model.
initial_learning_rate = 0.01
#lr_schedule = keras.optimizers.schedules.ExponentialDecay(
 #   initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
#)
model_2.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate),
    metrics=["acc"],
)


# Train the model, doing validation at the end of each epoch
epochs = 200
history = model_2.fit(X_train,y_train,validation_split=0.2,
                    epochs=epochs,shuffle=True,
                    verbose=2, batch_size=16,
                    callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=20)])
 #%%
fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model_2.history.history[metric])
    ax[i].plot(model_2.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])
# %%
# evaluate the model
scores = model_2.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model_2.metrics_names[1], scores[1]*100))

#%%
# Evaluate the test performance of the tuned model Hyperband
eval_result = model_2.evaluate(X_test, y_test)
print("[test loss, test accuracy]:", eval_result)

from sklearn.metrics import roc_curve
y_pred_keras = model_2.predict(X_test).ravel()
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
model_2.save("model_keras_acc_909_auc_968.h5")
print("Saved model to disk")
# %%
