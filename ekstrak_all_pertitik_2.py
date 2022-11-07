# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:23:10 2022

@author: nurde
"""

import numpy as np
import itertools
import random
from numpy import savez_compressed
from numpy import load
import pandas as pd

import os

os.chdir("C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616" )
os.getcwd()

df = pd.DataFrame(columns = ['Index', 'Lat', 'Lon', 'Hs', 'aspect', 'slope', 'TPI', 'TRI',
                             'elevasi', 'jalan','pemukiman', 'peat', 'ndvi',
                             'prec', 'hth'])
print(df)
for i in range(14,19,1):
    for j in range(0,2,1):
        df1= pd.read_csv(f'pertitik_20{i}_{j}.csv')
        df1.columns=df.columns
        df = pd.concat([df,df1], axis=0, ignore_index=True)
        print(f'pertitik_20{i}_{j}')




df.to_csv('train_pertitik.csv')
print('Done')