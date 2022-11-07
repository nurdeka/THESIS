# -*- coding: utf-8 -*-
#"""
#Created on Sun May 15 17:06:15 2022

#@author: deka
#"""

import numpy as np
import itertools
import random
from numpy import savez_compressed
from numpy import load
import pandas as pd

import os

### EXECUTE ####
## n 14 0 - 1908
#dict_data = load('data_0_1426.npz')
#data = dict_data['arr_0']
os.chdir("C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616" )
os.getcwd()


n= 14
hs = 0

dict_data = load(f'array_{n}.npz')

#data = dict_data[f"data_{n}"]  
data = dict_data["arr_0"]  
#generate random numbers based on x, y to list
a = np.random.choice(np.arange(0,data.shape[0]), data.shape[0], replace=False)
b = np.random.choice(np.arange(0,data.shape[1]), data.shape[1], replace=False)
#generate tuple combination a , b
c = random.sample(set(itertools.product(a,b)), len(a)*len(b))
# convert list of tuples to list of list
e = [list(ele) for ele in c]
#generate result randomly
#count = np.count_nonzero(data_{hs}4[0] == 1)
result = []
latlon = []
#a_list = range(-2,3,1)
#an_array = np.array(a_list)
for i in range(len(e)):
    if data[e[i][0],e[i][1],0]==hs:
        print('ok'+str(i))
        x = e[i][0]
        print('x='+str(e[i][0]))
        y = e[i][1]
        print('y='+str(e[i][1]))
        latlon.append([x,y])    

        result.append(data[x,y,:])
        
        print('nilai tengah='+str(data[e[i][0],e[i][1],0]))

result1 = pd.DataFrame(np.stack(result))
latlon1=pd.DataFrame(np.stack(latlon))
hasil4 = pd.concat([latlon1, result1], axis=1)
hasil4.to_csv(f'pertitik_{n}_{hs}.csv')
























#2015
n= 15 
data = dict_data[f"data_{n}"]  
#generate random numbers based on x, y to list
a = np.random.choice(np.arange(0,data.shape[1]), data.shape[1], replace=False)
b = np.random.choice(np.arange(0,data.shape[2]), data.shape[2], replace=False)
#generate tuple combination a , b
c = random.sample(set(itertools.product(a,b)), len(a)*len(b))
# convert list of tuples to list of list
e = [list(ele) for ele in c]
#generate result randomly
#count = np.count_nonzero(data_{hs}4[0] == 1)
result = []
latlon = []
#a_list = range(-2,3,1)
#an_array = np.array(a_list)
for i in range(len(e)):
    if data[0,e[i][0],e[i][1]]==hs:
        print('ok'+str(i))
        x = e[i][0]
        print('x='+str(e[i][0]))
        y = e[i][1]
        print('y='+str(e[i][1]))
        latlon.append([x,y])    

        result.append(data[:,x,y])
        
        print('nilai tengah='+str(data[0,e[i][0],e[i][1]]))

result1 = pd.DataFrame(np.stack(result))
latlon1=pd.DataFrame(np.stack(latlon))
hasil15 = pd.concat([latlon1, result1], axis=1)

#2016
n= 16 
data = dict_data[f"data_{n}"]  
#generate random numbers based on x, y to list
a = np.random.choice(np.arange(0,data.shape[1]), data.shape[1], replace=False)
b = np.random.choice(np.arange(0,data.shape[2]), data.shape[2], replace=False)
#generate tuple combination a , b
c = random.sample(set(itertools.product(a,b)), len(a)*len(b))
# convert list of tuples to list of list
e = [list(ele) for ele in c]
#generate result randomly
#count = np.count_nonzero(data_{hs}4[0] == 1)
result = []
latlon = []
#a_list = range(-2,3,1)
#an_array = np.array(a_list)
for i in range(len(e)):
    if data[0,e[i][0],e[i][1]]==hs:
        print('ok'+str(i))
        x = e[i][0]
        print('x='+str(e[i][0]))
        y = e[i][1]
        print('y='+str(e[i][1]))
        latlon.append([x,y])    

        result.append(data[:,x,y])
        
        print('nilai tengah='+str(data[0,e[i][0],e[i][1]]))

result1 = pd.DataFrame(np.stack(result))
latlon1=pd.DataFrame(np.stack(latlon))
hasil16 = pd.concat([latlon1, result1], axis=1)

#2017
n= 17
data = dict_data[f"data_{n}"]  
#generate random numbers based on x, y to list
a = np.random.choice(np.arange(0,data.shape[1]), data.shape[1], replace=False)
b = np.random.choice(np.arange(0,data.shape[2]), data.shape[2], replace=False)
#generate tuple combination a , b
c = random.sample(set(itertools.product(a,b)), len(a)*len(b))
# convert list of tuples to list of list
e = [list(ele) for ele in c]
#generate result randomly
#count = np.count_nonzero(data_{hs}4[0] == 1)
result = []
latlon = []
#a_list = range(-2,3,1)
#an_array = np.array(a_list)
for i in range(len(e)):
    if data[0,e[i][0],e[i][1]]==hs:
        print('ok'+str(i))
        x = e[i][0]
        print('x='+str(e[i][0]))
        y = e[i][1]
        print('y='+str(e[i][1]))
        latlon.append([x,y])    

        result.append(data[:,x,y])
        
        print('nilai tengah='+str(data[0,e[i][0],e[i][1]]))

result1 = pd.DataFrame(np.stack(result))
latlon1=pd.DataFrame(np.stack(latlon))
hasil17 = pd.concat([latlon1, result1], axis=1)

#2018
n= 18
data = dict_data[f"data_{n}"]  
#generate random numbers based on x, y to list
a = np.random.choice(np.arange(0,data.shape[1]), data.shape[1], replace=False)
b = np.random.choice(np.arange(0,data.shape[2]), data.shape[2], replace=False)
#generate tuple combination a , b
c = random.sample(set(itertools.product(a,b)), len(a)*len(b))
# convert list of tuples to list of list
e = [list(ele) for ele in c]
#generate result randomly
#count = np.count_nonzero(data_{hs}4[0] == 1)
result = []
latlon = []
#a_list = range(-2,3,1)
#an_array = np.array(a_list)
for i in range(len(e)):
    if data[0,e[i][0],e[i][1]]==hs:
        print('ok'+str(i))
        x = e[i][0]
        print('x='+str(e[i][0]))
        y = e[i][1]
        print('y='+str(e[i][1]))
        latlon.append([x,y])    

        result.append(data[:,x,y])
        
        print('nilai tengah='+str(data[0,e[i][0],e[i][1]]))

result1 = pd.DataFrame(np.stack(result))
latlon1=pd.DataFrame(np.stack(latlon))
hasil18 = pd.concat([latlon1, result1], axis=1)

#2019
n= 19 
data = dict_data[f"data_{n}"]  
#generate random numbers based on x, y to list
a = np.random.choice(np.arange(0,data.shape[1]), data.shape[1], replace=False)
b = np.random.choice(np.arange(0,data.shape[2]), data.shape[2], replace=False)
#generate tuple combination a , b
c = random.sample(set(itertools.product(a,b)), len(a)*len(b))
# convert list of tuples to list of list
e = [list(ele) for ele in c]
#generate result randomly
#count = np.count_nonzero(data_{hs}4[0] == 1)
result = []
latlon = []
#a_list = range(-2,3,1)
#an_array = np.array(a_list)
for i in range(len(e)):
    if data[0,e[i][0],e[i][1]]==hs:
        print('ok'+str(i))
        x = e[i][0]
        print('x='+str(e[i][0]))
        y = e[i][1]
        print('y='+str(e[i][1]))
        latlon.append([x,y])    

        result.append(data[:,x,y])
        
        print('nilai tengah='+str(data[0,e[i][0],e[i][1]]))

result1 = pd.DataFrame(np.stack(result))
latlon1=pd.DataFrame(np.stack(latlon))
hasil19 = pd.concat([latlon1, result1], axis=1)

#=============================================#
output_dir = ("F:/THESIS/CNN/DATA/POINT/out.csv" )

#train_frames = [hasil4, hasil15, hasil16, hasil17, hasil18]

#### DATA NOL  ###=============================================================
hs = 0

n= 14
data = dict_data[f"data_{n}"]  
#generate random numbers based on x, y to list
a = np.random.choice(np.arange(0,data.shape[1]), data.shape[1], replace=False)
b = np.random.choice(np.arange(0,data.shape[2]), data.shape[2], replace=False)
#generate tuple combination a , b
c = random.sample(set(itertools.product(a,b)), len(a)*len(b))
# convert list of tuples to list of list
e = [list(ele) for ele in c]
#generate result randomly
#count = np.count_nonzero(data_{hs}4[0] == 1)
result = []
latlon = []
#a_list = range(-2,3,1)
#an_array = np.array(a_list)
for i in range(20000):
    if data[0,e[i][0],e[i][1]]==hs:
        print('ok'+str(i))
        x = e[i][0]
        print('x='+str(e[i][0]))
        y = e[i][1]
        print('y='+str(e[i][1]))
        latlon.append([x,y])    

        result.append(data[:,x,y])
        
        print('nilai tengah='+str(data[0,e[i][0],e[i][1]]))

result1 = pd.DataFrame(np.stack(result[:1911]))
latlon1=pd.DataFrame(np.stack(latlon[:1911]))
hasil14_0 = pd.concat([latlon1, result1], axis=1)

#2015
n= 15 
data = dict_data[f"data_{n}"]  
#generate random numbers based on x, y to list
a = np.random.choice(np.arange(0,data.shape[1]), data.shape[1], replace=False)
b = np.random.choice(np.arange(0,data.shape[2]), data.shape[2], replace=False)
#generate tuple combination a , b
c = random.sample(set(itertools.product(a,b)), len(a)*len(b))
# convert list of tuples to list of list
e = [list(ele) for ele in c]
#generate result randomly
#count = np.count_nonzero(data_{hs}4[0] == 1)
result = []
latlon = []
#a_list = range(-2,3,1)
#an_array = np.array(a_list)
for i in range(20000):
    if data[0,e[i][0],e[i][1]]==hs:
        print('ok'+str(i))
        x = e[i][0]
        print('x='+str(e[i][0]))
        y = e[i][1]
        print('y='+str(e[i][1]))
        latlon.append([x,y])    

        result.append(data[:,x,y])
        
        print('nilai tengah='+str(data[0,e[i][0],e[i][1]]))

result1 = pd.DataFrame(np.stack(result[:1911]))
latlon1=pd.DataFrame(np.stack(latlon[:1911]))
hasil15_0 = pd.concat([latlon1, result1], axis=1)

#2016
n= 16 
data = dict_data[f"data_{n}"]  
#generate random numbers based on x, y to list
a = np.random.choice(np.arange(0,data.shape[1]), data.shape[1], replace=False)
b = np.random.choice(np.arange(0,data.shape[2]), data.shape[2], replace=False)
#generate tuple combination a , b
c = random.sample(set(itertools.product(a,b)), len(a)*len(b))
# convert list of tuples to list of list
e = [list(ele) for ele in c]
#generate result randomly
#count = np.count_nonzero(data_{hs}4[0] == 1)
result = []
latlon = []
#a_list = range(-2,3,1)
#an_array = np.array(a_list)
for i in range(20000):
    if data[0,e[i][0],e[i][1]]==hs:
        print('ok'+str(i))
        x = e[i][0]
        print('x='+str(e[i][0]))
        y = e[i][1]
        print('y='+str(e[i][1]))
        latlon.append([x,y])    

        result.append(data[:,x,y])
        
        print('nilai tengah='+str(data[0,e[i][0],e[i][1]]))

result1 = pd.DataFrame(np.stack(result[:1911]))
latlon1=pd.DataFrame(np.stack(latlon[:1911]))
hasil16_0 = pd.concat([latlon1, result1], axis=1)

#2017
n= 17
data = dict_data[f"data_{n}"]  
#generate random numbers based on x, y to list
a = np.random.choice(np.arange(0,data.shape[1]), data.shape[1], replace=False)
b = np.random.choice(np.arange(0,data.shape[2]), data.shape[2], replace=False)
#generate tuple combination a , b
c = random.sample(set(itertools.product(a,b)), len(a)*len(b))
# convert list of tuples to list of list
e = [list(ele) for ele in c]
#generate result randomly
#count = np.count_nonzero(data_{hs}4[0] == 1)
result = []
latlon = []
#a_list = range(-2,3,1)
#an_array = np.array(a_list)
for i in range(20000):
    if data[0,e[i][0],e[i][1]]==hs:
        print('ok'+str(i))
        x = e[i][0]
        print('x='+str(e[i][0]))
        y = e[i][1]
        print('y='+str(e[i][1]))
        latlon.append([x,y])    

        result.append(data[:,x,y])
        
        print('nilai tengah='+str(data[0,e[i][0],e[i][1]]))

result1 = pd.DataFrame(np.stack(result[:1911]))
latlon1=pd.DataFrame(np.stack(latlon[:1911]))
hasil17_0 = pd.concat([latlon1, result1], axis=1)

#2018
n= 18
data = dict_data[f"data_{n}"]  
#generate random numbers based on x, y to list
a = np.random.choice(np.arange(0,data.shape[1]), data.shape[1], replace=False)
b = np.random.choice(np.arange(0,data.shape[2]), data.shape[2], replace=False)
#generate tuple combination a , b
c = random.sample(set(itertools.product(a,b)), len(a)*len(b))
# convert list of tuples to list of list
e = [list(ele) for ele in c]
#generate result randomly
#count = np.count_nonzero(data_{hs}4[0] == 1)
result = []
latlon = []
#a_list = range(-2,3,1)
#an_array = np.array(a_list)
for i in range(20000):
    if data[0,e[i][0],e[i][1]]==hs:
        print('ok'+str(i))
        x = e[i][0]
        print('x='+str(e[i][0]))
        y = e[i][1]
        print('y='+str(e[i][1]))
        latlon.append([x,y])    

        result.append(data[:,x,y])
        
        print('nilai tengah='+str(data[0,e[i][0],e[i][1]]))

result1 = pd.DataFrame(np.stack(result[:1911]))
latlon1=pd.DataFrame(np.stack(latlon[:1911]))
hasil18_0 = pd.concat([latlon1, result1], axis=1)

#2019
n= 19 
data = dict_data[f"data_{n}"]  
#generate random numbers based on x, y to list
a = np.random.choice(np.arange(0,data.shape[1]), data.shape[1], replace=False)
b = np.random.choice(np.arange(0,data.shape[2]), data.shape[2], replace=False)
#generate tuple combination a , b
c = random.sample(set(itertools.product(a,b)), len(a)*len(b))
# convert list of tuples to list of list
e = [list(ele) for ele in c]
#generate result randomly
#count = np.count_nonzero(data_{hs}4[0] == 1)
result = []
latlon = []
#a_list = range(-2,3,1)
#an_array = np.array(a_list)
for i in range(20000):
    if data[0,e[i][0],e[i][1]]==hs:
        print('ok'+str(i))
        x = e[i][0]
        print('x='+str(e[i][0]))
        y = e[i][1]
        print('y='+str(e[i][1]))
        latlon.append([x,y])    

        result.append(data[:,x,y])
        
        print('nilai tengah='+str(data[0,e[i][0],e[i][1]]))

result1 = pd.DataFrame(np.stack(result[:1911]))
latlon1=pd.DataFrame(np.stack(latlon[:1911]))
hasil19_0 = pd.concat([latlon1, result1], axis=1)

#=============================================#
output_dir = ("F:/THESIS/CNN/DATA/POINT/out_19.csv" )

train_frames = [hasil19, hasil19_0]

df_train = pd.concat(train_frames, ignore_index=True, sort=False)
df_train.columns = ['lat', 'lon', 'hs', 'aspek',
                    'elevasi', 'jalan',
                    'peat', 'pemukiman',
                    'slope', 'TPI', 'TRI',
                    'NDVI', 'hujan', 'hth']

df_train.to_csv(output_dir)  


df_train = pd.concat(train_frames, ignore_index=True, sort=False)
df_train.columns = ['lat', 'lon', 'hs', 'aspek',
                    'elevasi', 'jalan',
                    'peat', 'pemukiman',
                    'slope', 'TPI', 'TRI',
                    'NDVI', 'hujan', 'hth']

df_train.to_csv(output_dir)  
