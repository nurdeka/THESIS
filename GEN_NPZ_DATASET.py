# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:23:54 2022

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
os.chdir("C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616" )
os.getcwd()
#%% TRAINING DATASET
print("============================= Generating Training Dataset ==========================")

roi = 15

print('ROI = ', roi)

def read_file(filepath):
    """Read and load volume"""
    # Read file
    dict_scan = load(filepath)
    # Get raw data
    scan = dict_scan['arr_0']
    return scan

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_file(path)
    # Normalize
    #volume = normalize(volume)
    # Resize width, height and depth
    #volume = resize_volume(volume)
    return volume

    # Folder "CT-0" consist of CT scans having normal lung tissue,
# no CT-signs of viral pneumonia.
normal_scan_paths = [
    os.path.join(os.getcwd(), f"TRAIN_OK\\DATA_0\\ALL_{roi}", x)
    for x in os.listdir(f"TRAIN_OK\\DATA_0\\ALL_{roi}")
]
# Folder "CT-23" consist of CT scans having several ground-glass opacifications,
# involvement of lung parenchyma.
abnormal_scan_paths = [
    os.path.join(os.getcwd(), f"TRAIN_OK\\DATA_1\\ALL_{roi}", x)
    for x in os.listdir(f"TRAIN_OK\\DATA_1\\ALL_{roi}")
]

print("HS scans with normal cases: " + str(len(normal_scan_paths)))
print("HS scans with abnormal cases: " + str(len(abnormal_scan_paths)))

# Read and process the scans.
# Each scan is resized across height, width, and depth and rescaled.
abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
normal_scans = np.array([process_scan(path) for path in normal_scan_paths])

print('normal scan shape :', normal_scans.shape)
print('abnormal scan shape :', abnormal_scans.shape)

normal_scans_nona = normal_scans[~np.isnan(normal_scans).any(axis=(-1,-2, -3))]
print('normal scan nona shape :', normal_scans_nona.shape)
abnormal_scans_nona = abnormal_scans[~np.isnan(abnormal_scans).any(axis=(-1,-2, -3))]
print('abnormal scan nona shape :', abnormal_scans_nona.shape)


# For the CT scans having presence of viral pneumonia
# assign 1, for the normal ones assign 0.
abnormal_labels = np.array([1 for _ in range(len(abnormal_scans_nona))])
normal_labels = np.array([0 for _ in range(len(normal_scans_nona))])

X_train = np.concatenate((abnormal_scans_nona, normal_scans_nona), axis=0)
y_train = np.concatenate((abnormal_labels, normal_labels), axis=0)
print('X_train shape : ', X_train.shape)
print('y_train shape : ',y_train.shape)

#%%t TESTING DATASET
print("============================= Generating Testing Dataset ==========================")
from numpy import load
def read_file(filepath):
    """Read and load volume"""
    # Read file
    dict_scan = load(filepath)
    # Get raw data
    scan = dict_scan['arr_0']
    return scan

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_file(path)
    # Normalize
    #volume = normalize(volume)
    # Resize width, height and depth
    #volume = resize_volume(volume)
    return volume

    # Folder "CT-0" consist of CT scans having normal lung tissue,
# no CT-signs of viral pneumonia.
normal_scan_paths = [
    os.path.join(os.getcwd(), f"TEST_OK\\DATA_0\\ALL_{roi}", x)
    for x in os.listdir(f"TEST_OK\\DATA_0\\ALL_{roi}")
]
# Folder "CT-23" consist of CT scans having several ground-glass opacifications,
# involvement of lung parenchyma.
abnormal_scan_paths = [
    os.path.join(os.getcwd(), f"TEST_OK\\DATA_1\\ALL_{roi}", x)
    for x in os.listdir(f"TEST_OK\\DATA_1\\ALL_{roi}")
]

print("HS scans with normal cases: " + str(len(normal_scan_paths)))
print("HS scans with abnormal cases: " + str(len(abnormal_scan_paths)))

# Read and process the scans.
# Each scan is resized across height, width, and depth and rescaled.
abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
normal_scans = np.array([process_scan(path) for path in normal_scan_paths])

print('normal scan shape :', normal_scans.shape)
print('abnormal scan shape :', abnormal_scans.shape)

#normal_scans_nona = normal_scans[~np.isnan(normal_scans).any(axis=(-1,-2, -3))]
normal_scans_nona = np.nan_to_num(normal_scans)
print('normal scan nona shape :', normal_scans_nona.shape)
#abnormal_scans_nona = abnormal_scans[~np.isnan(abnormal_scans).any(axis=(-1,-2, -3))]
abnormal_scans_nona = np.nan_to_num(abnormal_scans)
print('abnormal scan nona shape :', abnormal_scans_nona.shape)


# For the CT scans having presence of viral pneumonia
# assign 1, for the normal ones assign 0.
abnormal_labels = np.array([1 for _ in range(len(abnormal_scans_nona))])
normal_labels = np.array([0 for _ in range(len(normal_scans_nona))])

X_test = np.concatenate((abnormal_scans_nona, normal_scans_nona), axis=0)
y_test = np.concatenate((abnormal_labels, normal_labels), axis=0)
print('X_test shape : ', X_test.shape)
print('y_test shape : ', y_test.shape)
print('======================================================================')
#%% IMPORT NPZ
print('Saving on progress..')
np.savez_compressed(f'dataset_{roi}x{roi}.npz', X_train = X_train, y_train = y_train, 
                    X_test=X_test, y_test=y_test)

print(f'Saving complete on : {roi}x{roi} datasets')

#%% LOADING NPZ files


dict_data = load(f'dataset_{roi}x{roi}.npz')
dict_data.files
#data = dict_data["arr_0"]
#data.shape