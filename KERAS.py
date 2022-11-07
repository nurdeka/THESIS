#%% 
# KERAS-TUNER
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
# %%
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

#%%
window = X_train.shape[1]
channel = X_train.shape[3]
print(window)
print(channel)

#%%
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_val.shape)

#%%
def build_model(hp):
    # create model object
    model = keras.Sequential([
    #adding first convolutional layer    
    keras.layers.Conv3D(
        #adding filter 
        filters=hp.Int('conv_1_filter', min_value=16, max_value=256, step=16),
        # adding filter size or kernel size
        kernel_size=hp.Choice('conv_1_kernel', values = [2,3]),
        #activation function
        activation='relu', padding = 'same',
        input_shape=(window,window,channel,1)),

    keras.layers.MaxPooling3D(pool_size=2),
    #hp_dropout = hp.Float('rate', min_value=0.1, max_value=0.5, step=0.1),
    keras.layers.Dropout(
        rate=hp.Float('rate_1', min_value=0.0, max_value=0.5, step=0.1)),
    # adding second convolutional layer 
#    keras.layers.Conv3D(
        #adding filter 
#        filters=hp.Int('conv_2_filter', min_value=16, max_value=128, step=16),
        #adding filter size or kernel size
#        kernel_size=hp.Choice('conv_2_kernel', values = [2,3]),
        #activation function
#        activation='relu', padding = 'same'
#    ),
#    keras.layers.MaxPooling3D(pool_size=2),
#    keras.layers.Dropout(
#        rate=hp.Float('rate_2', min_value=0.0, max_value=0.5, step=0.1)),
    # adding flatten layer    
    keras.layers.Flatten(),
    # adding dense layer    
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=16, max_value=256, step=16),
        activation='relu'
    ),
    keras.layers.Dropout(
        rate=hp.Float('rate_3', min_value=0.0, max_value=0.5, step=0.1)),
    keras.layers.Dense(
        units=hp.Int('dense_2_units', min_value=16, max_value=256, step=16),
        activation='relu'
    ),
    keras.layers.Dropout(
        rate=hp.Float('rate_4', min_value=0.0, max_value=0.5, step=0.1)),
    # output layer    
    keras.layers.Dense(
        units=hp.Choice('output', values = [1]), activation='sigmoid')
    ])
    #compilation of model
    #model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
    #          loss='sparse_categorical_crossentropy',
    #          metrics=['accuracy'])
    
      # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy',metrics=['accuracy'])
    return model

import keras_tuner
# Initialize the `HyperParameters` and set the values.
hp = keras_tuner.HyperParameters()
#hp.values["model_type"] = "cnn"
# Build the model using the `HyperParameters`.
model = build_model(hp)
# Test if the model runs with our data.
#model(X_train)
# Print a summary of the model.
model.summary()
# %%
#importing random search
import keras_tuner
import tensorflow as tf
import numpy as np
from keras_tuner import RandomSearch, Hyperband
#creating randomsearch object
tuner = Hyperband(build_model,
                  objective='val_accuracy',
                  max_epochs=200,
                  factor=3,
                  directory='keras_tuner',
                  project_name = 'tes_20220609',
                  overwrite=True)
# search best parameter
tuner.search(X_train, y_train,epochs=200,validation_data=[X_val, y_val], callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=10)])
# %%
