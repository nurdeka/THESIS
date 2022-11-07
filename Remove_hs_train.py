# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 07:25:28 2022

@author: nurde
"""

import numpy as np
import itertools
import random
from numpy import savez_compressed
from numpy import load

import os

#MENGHAPUS LAYER 1 (HOTSPOT)
hs = 0
r =5
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST_OK\\DATA_{hs}\\ALL_{r}\\')
os.getcwd()
os.chdir(f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST\\DATA_{hs}\\ALL_{r}\\')
for i in range(30541):
    dict_data = load(f"data_{i}.npz")
    data = dict_data['arr_0']
    data = data[:,:,1:]
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), data)
    print('done at data '+str(i))


hs = 1
r =5
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST_OK\\DATA_{hs}\\ALL_{r}\\')
os.getcwd()
os.chdir(f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST\\DATA_{hs}\\ALL_{r}\\')
for i in range(29842):
    dict_data = load(f"data_{i}.npz")
    data = dict_data['arr_0']
    data = data[:,:,1:]
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), data)
    print('done at data '+str(i))
  
hs = 0
r =7
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST_OK\\DATA_{hs}\\ALL_{r}\\')
os.getcwd()
os.chdir(f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST\\DATA_{hs}\\ALL_{r}\\')
for i in range(30541):
    dict_data = load(f"data_{i}.npz")
    data = dict_data['arr_0']
    data = data[:,:,1:]
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), data)
    print('done at data '+str(i))

hs = 1
r =7
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST_OK\\DATA_{hs}\\ALL_{r}\\')
os.getcwd()
os.chdir(f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST\\DATA_{hs}\\ALL_{r}\\')
for i in range(29831):
    dict_data = load(f"data_{i}.npz")
    data = dict_data['arr_0']
    data = data[:,:,1:]
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), data)
    print('done at data '+str(i))
  
hs = 0
r =9
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST_OK\\DATA_{hs}\\ALL_{r}\\')
os.getcwd()
os.chdir(f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST\\DATA_{hs}\\ALL_{r}\\')
for i in range(30541):
    dict_data = load(f"data_{i}.npz")
    data = dict_data['arr_0']
    data = data[:,:,1:]
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), data)
    print('done at data '+str(i))
    
hs = 1
r =9
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST_OK\\DATA_{hs}\\ALL_{r}\\')
os.getcwd()
os.chdir(f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST\\DATA_{hs}\\ALL_{r}\\')
for i in range(29824):
    dict_data = load(f"data_{i}.npz")
    data = dict_data['arr_0']
    data = data[:,:,1:]
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), data)
    print('done at data '+str(i))
    
hs = 0
r =11
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST_OK\\DATA_{hs}\\ALL_{r}\\')
os.getcwd()
os.chdir(f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST\\DATA_{hs}\\ALL_{r}\\')
for i in range(30540):
    dict_data = load(f"data_{i}.npz")
    data = dict_data['arr_0']
    data = data[:,:,1:]
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), data)
    print('done at data '+str(i))
    
hs = 1
r =11
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST_OK\\DATA_{hs}\\ALL_{r}\\')
os.getcwd()
os.chdir(f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST\\DATA_{hs}\\ALL_{r}\\')
for i in range(29813):
    dict_data = load(f"data_{i}.npz")
    data = dict_data['arr_0']
    data = data[:,:,1:]
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), data)
    print('done at data '+str(i))

hs = 0
r =15
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST_OK\\DATA_{hs}\\ALL_{r}\\')
os.getcwd()
os.chdir(f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST\\DATA_{hs}\\ALL_{r}\\')
for i in range(30537):
    dict_data = load(f"data_{i}.npz")
    data = dict_data['arr_0']
    data = data[:,:,1:]
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), data)
    print('done at data '+str(i))
    
hs = 1
r =15
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST_OK\\DATA_{hs}\\ALL_{r}\\')
os.getcwd()
os.chdir(f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST\\DATA_{hs}\\ALL_{r}\\')
for i in range(29790):
    dict_data = load(f"data_{i}.npz")
    data = dict_data['arr_0']
    data = data[:,:,1:]
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), data)
    print('done at data '+str(i))
    
hs = 0
r =21
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST_OK\\DATA_{hs}\\ALL_{r}\\')
os.getcwd()
os.chdir(f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST\\DATA_{hs}\\ALL_{r}\\')
for i in range(30534):
    dict_data = load(f"data_{i}.npz")
    data = dict_data['arr_0']
    data = data[:,:,1:]
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), data)
    print('done at data '+str(i))
    
hs = 1
r =21
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST_OK\\DATA_{hs}\\ALL_{r}\\')
os.getcwd()
os.chdir(f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST\\DATA_{hs}\\ALL_{r}\\')
for i in range(29750):
    dict_data = load(f"data_{i}.npz")
    data = dict_data['arr_0']
    data = data[:,:,1:]
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), data)
    print('done at data '+str(i))
  
hs = 0
r =25
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST_OK\\DATA_{hs}\\ALL_{r}\\')
os.getcwd()
os.chdir(f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST\\DATA_{hs}\\ALL_{r}\\')
for i in range(30533):
    dict_data = load(f"data_{i}.npz")
    data = dict_data['arr_0']
    data = data[:,:,1:]
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), data)
    print('done at data '+str(i))
    
hs = 1
r =25
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST_OK\\DATA_{hs}\\ALL_{r}\\')
os.getcwd()
os.chdir(f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST\\DATA_{hs}\\ALL_{r}\\')
for i in range(29725):
    dict_data = load(f"data_{i}.npz")
    data = dict_data['arr_0']
    data = data[:,:,1:]
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), data)
    print('done at data '+str(i))
  
hs = 0
r =31
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST_OK\\DATA_{hs}\\ALL_{r}\\')
os.getcwd()
os.chdir(f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST\\DATA_{hs}\\ALL_{r}\\')
for i in range(30530):
    dict_data = load(f"data_{i}.npz")
    data = dict_data['arr_0']
    data = data[:,:,1:]
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), data)
    print('done at data '+str(i))
    
hs = 1
r =31
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST_OK\\DATA_{hs}\\ALL_{r}\\')
os.getcwd()
os.chdir(f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST\\DATA_{hs}\\ALL_{r}\\')
for i in range(29674):
    dict_data = load(f"data_{i}.npz")
    data = dict_data['arr_0']
    data = data[:,:,1:]
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), data)
    print('done at data '+str(i))
  
  
  