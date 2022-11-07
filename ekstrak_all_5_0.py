# -*- coding: utf-8 -*-
"""
Created on Sun May 15 17:06:15 2022

@author: deka
"""

import numpy as np
import itertools
import random
from numpy import savez_compressed
from numpy import load

import os

### EXECUTE ####
## n 14 0 - 1908

os.chdir("C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616" )
os.getcwd()
#%% 5x5 0-18270
n = 14
hs = 1
dict_data = load(f'array_{n}.npz')

data = dict_data["arr_0"]  
a = np.random.choice(np.arange(2,7463), 7461, replace=False)
b = np.random.choice(np.arange(2,2687), 2685, replace=False)
#generate tuple combination a , b
c = random.sample(set(itertools.product(a,b)), 20032785)
# convert list of tuples to list of list
e = [list(ele) for ele in c]
#generate result randomly
#count = np.count_nonzero(data_{hs}4[0] == 1)
result = []
latlon = []
#a_list = range(-2,3,1)
an_array = np.array(a_list)
for i in range(len(e)):
    if data[e[i][0],e[i][1],0]==hs:
        print('ok'+str(i))
        x = e[i][0]
        print('x='+str(e[i][0]))
        y = e[i][1]
        print('y='+str(e[i][1]))
        latlon.append([x,y])    
        idx_1 = an_array + x
        idx = idx_1.tolist()
        idy_1 = an_array + y
        idy = idy_1.tolist()
        ixgrid = np.ix_(idx,idy)
        result.append(data[ixgrid[0],ixgrid[1],:])
        
        print('nilai tengah='+str(data[e[i][0],e[i][1],0]))

latlon1 = []
result1 = []
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TRAIN\\DATA_{hs}\\ALL_5\\')

for i, j in zip(range(0,0+len(result)), range(len(result))):
    #print(f'coba{i}',f'dan{j}')
    #os.path.join(output_dir, savez_compressed(f"data_{i}.npz", result[i]))
    #savez_compressed(os.path.join(output_dir,f"data_{i}.npz"), result[j])
    latlon1.append(latlon[i]) 
    latlon1.append(latlon[j]) 
    print('done '+str(i)) 
savez_compressed(os.path.join(output_dir,f"latlon_{hs}_{n}.npz"), latlon1)
x = len(result)

# 5x5 0-18270
n = 15
hs = 1
dict_data = load(f'array_{n}.npz')

data = dict_data["arr_0"]  
a = np.random.choice(np.arange(2,7463), 7461, replace=False)
b = np.random.choice(np.arange(2,2687), 2685, replace=False)
#generate tuple combination a , b
c = random.sample(set(itertools.product(a,b)), 20032785)
# convert list of tuples to list of list
e = [list(ele) for ele in c]
#generate result randomly
#count = np.count_nonzero(data_{hs}4[0] == 1)
result = []
latlon = []
a_list = range(-2,3,1)
an_array = np.array(a_list)
for i in range(len(e)):
    if data[e[i][0],e[i][1],0]==hs:
        print('ok'+str(i))
        x = e[i][0]
        print('x='+str(e[i][0]))
        y = e[i][1]
        print('y='+str(e[i][1]))
        latlon.append([x,y])    
        idx_1 = an_array + x
        idx = idx_1.tolist()
        idy_1 = an_array + y
        idy = idy_1.tolist()
        ixgrid = np.ix_(idx,idy)
        result.append(data[ixgrid[0],ixgrid[1],:])
        
        print('nilai tengah='+str(data[e[i][0],e[i][1],0]))

latlon1 = []
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TRAIN\\DATA_{hs}\\ALL_5\\')

for i, j in zip(range(z, z+len(result)), range(len(result))):
    #print(f'coba{i}',f'dan{j}')
    #os.path.join(output_dir, savez_compressed(f"data_{i}.npz", result[i]))
    savez_compressed(os.path.join(output_dir,f"data_{i}.npz"), result[j])
    latlon1.append(latlon[j]) 
    print('done '+str(i)) 
savez_compressed(os.path.join(output_dir,f"latlon_{hs}_{n}.npz"), latlon1)

z = z+len(result)
 #%%
## n 16 7854-7864 (10)
n= 16
data = dict_data[f"data_{n}"]  
#generate random numbers based on x, y to list
a = np.random.choice(np.arange(2,7475), 7473, replace=False)
b = np.random.choice(np.arange(2,2699), 2697, replace=False)
#generate tuple combination a , b
c = random.sample(set(itertools.product(a,b)), 20154681)
# convert list of tuples to list of list
e = [list(ele) for ele in c]
#generate result randomly
#count = np.count_nonzero(data_{hs}4[0] == 1)
result = []
latlon = []
a_list = range(-2,3,1)
an_array = np.array(a_list)
for i in range(20000):
    if data[0,e[i][0],e[i][1]]==hs:
        print('ok'+str(i))
        x = e[i][0]
        print('x='+str(e[i][0]))
        y = e[i][1]
        print('y='+str(e[i][1]))
        latlon.append([x,y])    
        idx_1 = an_array + x
        idx = idx_1.tolist()
        idy_1 = an_array + y
        idy = idy_1.tolist()
        ixgrid = np.ix_(idx,idy)
        result.append(data[:,ixgrid[0],ixgrid[1]])
        
        print('nilai tengah='+str(data[0,e[i][0],e[i][1]]))

latlon1 = []
output_dir = ('F:/THESIS/CNN/DATA/ALL_5_0/')

for i, j in zip(range(7854,7864), range(10)):
    #print(f'coba{i}',f'dan{j}')
    #os.path.join(output_dir, savez_compressed(f"data_{i}.npz", result[i]))
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), result[j])
    latlon1.append(latlon[j]) 
    print('done '+str(i)) 
savez_compressed(os.path.join(output_dir,f"latlon_{n}.npz"), latlon1)

## n17 7864-7889 (25)
n= 17
data = dict_data[f"data_{n}"]  
#generate random numbers based on x, y to list
a = np.random.choice(np.arange(2,7475), 7473, replace=False)
b = np.random.choice(np.arange(2,2699), 2697, replace=False)
#generate tuple combination a , b
c = random.sample(set(itertools.product(a,b)), 20154681)
# convert list of tuples to list of list
e = [list(ele) for ele in c]
#generate result randomly
#count = np.count_nonzero(data_{hs}4[0] == 1)
result = []
latlon = []
a_list = range(-2,3,1)
an_array = np.array(a_list)
for i in range(20000):
    if data[0,e[i][0],e[i][1]]==hs:
        print('ok'+str(i))
        x = e[i][0]
        print('x='+str(e[i][0]))
        y = e[i][1]
        print('y='+str(e[i][1]))
        latlon.append([x,y])    
        idx_1 = an_array + x
        idx = idx_1.tolist()
        idy_1 = an_array + y
        idy = idy_1.tolist()
        ixgrid = np.ix_(idx,idy)
        result.append(data[:,ixgrid[0],ixgrid[1]])
        
        print('nilai tengah='+str(data[0,e[i][0],e[i][1]]))

latlon1 = []
output_dir = ('F:/THESIS/CNN/DATA/ALL_5_0/')

for i, j in zip(range(7864,7889), range(25)):
    #print(f'coba{i}',f'dan{j}')
    #os.path.join(output_dir, savez_compressed(f"data_{i}.npz", result[i]))
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), result[j])
    latlon1.append(latlon[j]) 
    print('done '+str(i))  
savez_compressed(os.path.join(output_dir,f"latlon_{n}.npz"), latlon1)

## n18 7889-8568 (679)
n= 18
data = dict_data[f"data_{n}"]  
#generate random numbers based on x, y to list
a = np.random.choice(np.arange(2,7475), 7473, replace=False)
b = np.random.choice(np.arange(2,2699), 2697, replace=False)
#generate tuple combination a , b
c = random.sample(set(itertools.product(a,b)), 20154681)
# convert list of tuples to list of list
e = [list(ele) for ele in c]
#generate result randomly
#count = np.count_nonzero(data_{hs}4[0] == 1)
result = []
latlon = []
a_list = range(-2,3,1)
an_array = np.array(a_list)
for i in range(20000):
    if data[0,e[i][0],e[i][1]]==hs:
        print('ok'+str(i))
        x = e[i][0]
        print('x='+str(e[i][0]))
        y = e[i][1]
        print('y='+str(e[i][1]))
        latlon.append([x,y])    
        idx_1 = an_array + x
        idx = idx_1.tolist()
        idy_1 = an_array + y
        idy = idy_1.tolist()
        ixgrid = np.ix_(idx,idy)
        result.append(data[:,ixgrid[0],ixgrid[1]])
        
        print('nilai tengah='+str(data[0,e[i][0],e[i][1]]))

latlon1 = []
output_dir = ('F:/THESIS/CNN/DATA/ALL_5_0/')

for i, j in zip(range(7889, 8568), range(679)):
    #print(f'coba{i}',f'dan{j}')
    #os.path.join(output_dir, savez_compressed(f"data_{i}.npz", result[i]))
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), result[j])
    latlon1.append(latlon[j]) 
    print('done '+str(i)) 
savez_compressed(os.path.join(output_dir,f"latlon_{n}.npz"), latlon1)

## n19 8568-11974 (3406)
n= 19
data = dict_data[f"data_{n}"]  
#generate random numbers based on x, y to list
a = np.random.choice(np.arange(2,7475), 7473, replace=False)
b = np.random.choice(np.arange(2,2699), 2697, replace=False)
#generate tuple combination a , b
c = random.sample(set(itertools.product(a,b)), 20154681)
# convert list of tuples to list of list
e = [list(ele) for ele in c]
#generate result randomly
#count = np.count_nonzero(data_{hs}4[0] == 1)
result = []
latlon = []
a_list = range(-2,3,1)
an_array = np.array(a_list)
for i in range(20000):
    if data[0,e[i][0],e[i][1]]==hs:
        print('ok'+str(i))
        x = e[i][0]
        print('x='+str(e[i][0]))
        y = e[i][1]
        print('y='+str(e[i][1]))
        latlon.append([x,y])    
        idx_1 = an_array + x
        idx = idx_1.tolist()
        idy_1 = an_array + y
        idy = idy_1.tolist()
        ixgrid = np.ix_(idx,idy)
        result.append(data[:,ixgrid[0],ixgrid[1]])
        
        print('nilai tengah='+str(data[0,e[i][0],e[i][1]]))

latlon1 = []
output_dir = ('F:/THESIS/CNN/DATA/ALL_5_0/')

for i, j in zip(range(8568, 11971), range(3403)):
    #print(f'coba{i}',f'dan{j}')
    #os.path.join(output_dir, savez_compressed(f"data_{i}.npz", result[i]))
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), result[j])
    latlon1.append(latlon[j]) 
    print('done '+str(i)) 
savez_compressed(os.path.join(output_dir,f"latlon_{n}.npz"), latlon1)


#MENGHAPUS LAYER 1 (HOTSPOT)
output_dir = ('F:/THESIS/CNN/DATA/ALL_5_0/')
os.getcwd()
os.chdir('F:/THESIS/CNN/DATA/ALL_5_0/')
for i in range(11971):
    dict_data = load(f"data_{hs}_{i}.npz")
    data = dict_data['arr_0']
    data = data[1:,:,:]
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), data)
    print('done at data '+str(i))
    
# MENGGANTI DIMENSI
for i in range(11971):
    dict_data = load(f"data_{hs}_{i}.npz")
    data = dict_data['arr_0']
    data = np.moveaxis(data, 0, -1)
    savez_compressed(os.path.join(output_dir,f"data_{hs}_{i}.npz"), data)
    print('done at data '+str(i))