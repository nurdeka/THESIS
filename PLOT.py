# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 21:14:19 2022

@author: nurde
"""


import numpy as np 
import matplotlib.pyplot as plt 
  
X = ['Group A','Group B','Group C','Group D']
Ygirls = [10,20,20,40]
Zboys = [20,30,25,30]
  
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, Ygirls, 0.4, label = 'Girls')
plt.bar(X_axis + 0.2, Zboys, 0.4, label = 'Boys')
  
plt.xticks(X_axis, X)
plt.xlabel("Groups")
plt.ylabel("Number of Students")
plt.title("Number of Students in each group")
plt.legend()
plt.ylim(5,50)
plt.show()


#PLOT LINE PYTHON
import matplotlib.pyplot as plt
y = [0.887, 0.905, 0.902, 0.926, 0.929, 0.935]
x = ['5x5','7x7', '9x9', '11x11', '15x15', '25x25']
plt.figure(figsize=(6, 4), dpi=200)
plt.rcParams["font.family"] = "Times New Roman"

plt.plot(x,y, marker='o')
plt.ylim(0.88, 0.94)
plt.xlabel('ROI')
plt.ylabel('Akurasi')
plt.grid(axis= 'y')
plt.legend()
#plt.title('ROC curve')
plt.show()

#PLOT MULTIBAR

import numpy as np 
import matplotlib.pyplot as plt 
  
X = ['SVC','GBC','DNN','RF', 'CNN']
Ygirls = [0.88,0.88,0.88,0.89, 0.881]
Zboys = [0.89,0.89,0.89,0.90, 0.935]
  
X_axis = np.arange(len(X))

plt.figure(figsize=(6, 4), dpi=200)
plt.rcParams["font.family"] = "Times New Roman"
plt.bar(X_axis - 0.2, Ygirls, 0.4, label = 'Tanpa Gambut', color='tab:blue')
plt.bar(X_axis + 0.2, Zboys, 0.4, label = 'Dengan Gambut', color='tab:green')
plt.ylim(0.86, 0.94)
plt.xticks(X_axis, X)
plt.xlabel("Model")
plt.ylabel("Akurasi")
#plt.title("Number of Students in each group")
plt.legend()
plt.show()

#%%
#%%
from numpy import load
import os
import numpy as np

os.chdir('G:\\My Drive\\DATA\\MODEL')


dict_data = load('model9var_11_926_980_history.npz')
dict_data.files
data = dict_data[]
#joblib.dump(model1, filename1)
#joblib.dump(model2, filename2)
#joblib.dump(model3, filename3)
#loaded_model.load_weights(filename4)

 
# some time later...
 
# load the model from disk
path = os.path.join(os.getcwd(), filename1)
rf_model = joblib.load(path)