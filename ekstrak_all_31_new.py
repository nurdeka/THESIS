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

### EXECUTE ###
os.chdir("C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616" )
os.getcwd()
#%% 5x5 0-18270
n = 19
hs = 0
r = 15


va = int((r-1)/2)
wa = int((r-1)/2)
vb = int(7465 - va)
wb = int(2689 - wa)
vi = 7465 - (r-1)
wi = 2689 - (r-1)
z = 0
a_list = range(-va,va+1,1)

dict_data = load(f'array_{n}.npz')

data = dict_data["arr_0"]  
a = np.random.choice(np.arange(va,vb), vi, replace=False)
b = np.random.choice(np.arange(wa,wb), wi, replace=False)
#generate tuple combination a , b
c = random.sample(set(itertools.product(a,b)), vi*wi)
# convert list of tuples to list of list
e = [list(ele) for ele in c]
#generate result randomly
#count = np.count_nonzero(data_{hs}4[0] == 1)
result = []
latlon = []

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
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TEST\\DATA_{hs}\\ALL_{r}\\')

for i, j in zip(range(z,z+len(result)), range(len(result))):
    #print(f'coba{i}',f'dan{j}')
    #os.path.join(output_dir, savez_compressed(f"data_{i}.npz", result[i]))
    savez_compressed(os.path.join(output_dir,f"data_{i}.npz"), result[j])
    latlon1.append(latlon[j]) 
    print('done '+str(i)) 
savez_compressed(os.path.join(output_dir,f"latlon_{hs}_{n}.npz"), latlon1)
print('Done.....')

#%%
# 5x5 
n = 15
z = z+len(result)
dict_data = load(f'array_{n}.npz')

data = dict_data["arr_0"]  

result = []
latlon = []
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
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TRAIN\\DATA_{hs}\\ALL_{r}\\')

for i, j in zip(range(z, z+len(result)), range(len(result))):
    #print(f'coba{i}',f'dan{j}')
    #os.path.join(output_dir, savez_compressed(f"data_{i}.npz", result[i]))
    savez_compressed(os.path.join(output_dir,f"data_{i}.npz"), result[j])
    latlon1.append(latlon[j]) 
    print('done '+str(i)) 
savez_compressed(os.path.join(output_dir,f"latlon_{hs}_{n}.npz"), latlon1)

#%%
# 5x5 
n = 16
z = z+len(result)
dict_data = load(f'array_{n}.npz')

data = dict_data["arr_0"]  

result = []
latlon = []

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
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TRAIN\\DATA_{hs}\\ALL_{r}\\')

for i, j in zip(range(z, z+len(result)), range(len(result))):
    #print(f'coba{i}',f'dan{j}')
    #os.path.join(output_dir, savez_compressed(f"data_{i}.npz", result[i]))
    savez_compressed(os.path.join(output_dir,f"data_{i}.npz"), result[j])
    latlon1.append(latlon[j]) 
    print('done '+str(i)) 
savez_compressed(os.path.join(output_dir,f"latlon_{hs}_{n}.npz"), latlon1)


#%%
# 5x5 
n = 17
z = z+len(result)
dict_data = load(f'array_{n}.npz')

data = dict_data["arr_0"]  
result = []
latlon = []

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
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TRAIN\\DATA_{hs}\\ALL_{r}\\')

for i, j in zip(range(z, z+len(result)), range(len(result))):
    #print(f'coba{i}',f'dan{j}')
    #os.path.join(output_dir, savez_compressed(f"data_{i}.npz", result[i]))
    savez_compressed(os.path.join(output_dir,f"data_{i}.npz"), result[j])
    latlon1.append(latlon[j]) 
    print('done '+str(i)) 
savez_compressed(os.path.join(output_dir,f"latlon_{hs}_{n}.npz"), latlon1)

#%%
# 
n = 18
z = z+len(result)
dict_data = load(f'array_{n}.npz')

data = dict_data["arr_0"]  

result = []
latlon = []

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
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TRAIN\\DATA_{hs}\\ALL_{r}\\')

for i, j in zip(range(z, z+len(result)), range(len(result))):
    #print(f'coba{i}',f'dan{j}')
    #os.path.join(output_dir, savez_compressed(f"data_{i}.npz", result[i]))
    savez_compressed(os.path.join(output_dir,f"data_{i}.npz"), result[j])
    latlon1.append(latlon[j]) 
    print('done '+str(i)) 
savez_compressed(os.path.join(output_dir,f"latlon_{hs}_{n}.npz"), latlon1)


print('Done Part 1')

#%% 5x5 0-18270
n = 14
hs = 1
z = 0
dict_data = load(f'array_{n}.npz')

data = dict_data["arr_0"]  
#a = np.random.choice(np.arange(3,7462), 7459, replace=False)
#b = np.random.choice(np.arange(3,2686), 2683, replace=False)
#generate tuple combination a , b
#c = random.sample(set(itertools.product(a,b)), 20032785)
# convert list of tuples to list of list
#e = [list(ele) for ele in c]
#generate result randomly
#count = np.count_nonzero(data_{hs}4[0] == 1)
result = []
latlon = []

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
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TRAIN\\DATA_{hs}\\ALL_{r}\\')

for i, j in zip(range(z,z+len(result)), range(len(result))):
    #print(f'coba{i}',f'dan{j}')
    #os.path.join(output_dir, savez_compressed(f"data_{i}.npz", result[i]))
    savez_compressed(os.path.join(output_dir,f"data_{i}.npz"), result[j])
    latlon1.append(latlon[j]) 
    print('done '+str(i)) 
savez_compressed(os.path.join(output_dir,f"latlon_{hs}_{n}.npz"), latlon1)

#%%
# 5x5 
n = 15
z = z+len(result)
dict_data = load(f'array_{n}.npz')

data = dict_data["arr_0"]  

result = []
latlon = []

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
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TRAIN\\DATA_{hs}\\ALL_{r}\\')

for i, j in zip(range(z, z+len(result)), range(len(result))):
    #print(f'coba{i}',f'dan{j}')
    #os.path.join(output_dir, savez_compressed(f"data_{i}.npz", result[i]))
    savez_compressed(os.path.join(output_dir,f"data_{i}.npz"), result[j])
    latlon1.append(latlon[j]) 
    print('done '+str(i)) 
savez_compressed(os.path.join(output_dir,f"latlon_{hs}_{n}.npz"), latlon1)

#%%
# 5x5 
n = 16
z = z+len(result)
dict_data = load(f'array_{n}.npz')

data = dict_data["arr_0"]  

result = []
latlon = []

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
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TRAIN\\DATA_{hs}\\ALL_{r}\\')

for i, j in zip(range(z, z+len(result)), range(len(result))):
    #print(f'coba{i}',f'dan{j}')
    #os.path.join(output_dir, savez_compressed(f"data_{i}.npz", result[i]))
    savez_compressed(os.path.join(output_dir,f"data_{i}.npz"), result[j])
    latlon1.append(latlon[j]) 
    print('done '+str(i)) 
savez_compressed(os.path.join(output_dir,f"latlon_{hs}_{n}.npz"), latlon1)


#%%
# 5x5 
n = 17
z = z+len(result)
dict_data = load(f'array_{n}.npz')

data = dict_data["arr_0"]  
result = []
latlon = []

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
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TRAIN\\DATA_{hs}\\ALL_{r}\\')

for i, j in zip(range(z, z+len(result)), range(len(result))):
    #print(f'coba{i}',f'dan{j}')
    #os.path.join(output_dir, savez_compressed(f"data_{i}.npz", result[i]))
    savez_compressed(os.path.join(output_dir,f"data_{i}.npz"), result[j])
    latlon1.append(latlon[j]) 
    print('done '+str(i)) 
savez_compressed(os.path.join(output_dir,f"latlon_{hs}_{n}.npz"), latlon1)

#%%
# 
n = 18
z = z+len(result)
dict_data = load(f'array_{n}.npz')

data = dict_data["arr_0"]  

result = []
latlon = []

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
output_dir = (f'C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616\\TRAIN\\DATA_{hs}\\ALL_{r}\\')

for i, j in zip(range(z, z+len(result)), range(len(result))):
    #print(f'coba{i}',f'dan{j}')
    #os.path.join(output_dir, savez_compressed(f"data_{i}.npz", result[i]))
    savez_compressed(os.path.join(output_dir,f"data_{i}.npz"), result[j])
    latlon1.append(latlon[j]) 
    print('done '+str(i)) 
savez_compressed(os.path.join(output_dir,f"latlon_{hs}_{n}.npz"), latlon1)



print('Done All...!!!')




