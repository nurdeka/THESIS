# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 13:43:16 2022

@author: nurde
"""

import numpy as np
import itertools
import random
from numpy import savez_compressed
from numpy import load
import pandas as pd

import os

### EXECUTE ####
## n 14 0 - 1908

os.chdir("C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616" )
os.getcwd()

dict_data = load('C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST\\DATA_1\\ALL_5\\latlon_1_19.npz')
dict_data.files

latlon_1 = dict_data["arr_0"]

dict_data = load('C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST\\DATA_0\\ALL_5\\latlon_0_19.npz')
dict_data.files

latlon_0 = dict_data["arr_0"]

df = pd.read_csv('hs_grid_master.csv')

print(df.to_string()) 


#Ekstrak LATLON NUMBER ke DEGREE
lat_arr = np.arange(-3.467676, -1.228176, 0.000300000000000189)
lon_arr = np.arange(113.5779, 114.3844, 0.000300)


latlon_null = []
for i in range(len(latlon_0)):
    x = latlon_0[i][0]
    y = latlon_0[i][1]
    x1 = lat_arr[x]
    y1 = lon_arr[y]
    latlon_null.append([x1,y1])
    print(i)
    
latlon_ok =[]
for i in range(len(latlon_null)):
    latlon_ok.append(latlon_null[i])
    print(f'OK data ke : {i}')
