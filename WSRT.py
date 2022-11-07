#%%
#WSRT

import scipy.stats as stats
group_CNN = pred list[]
group_RF = pred list[]
group_GBC = pred list[]
group_SVC = pred list[]
group_NN = pred list[]
#perform the Wilcoxon-Signed Rank Test
print('CNN vs RF :', stats.wilcoxon(cn_probs, rf_probs[:,1]))
print('CNN vs GBC :', stats.wilcoxon(group_CNN, group_GBC))
print('CNN vs SVC :', stats.wilcoxon(group_CNN, group_SVC))
print('CNN vs NN :', stats.wilcoxon(group_CNN, group_NN))

from numpy import load
import os

os.chdir('G:\\My Drive\\DATA')
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

import joblib
import tensorflow
from tensorflow.keras.models import load_model
# save the model to disk
filename1 = 'finalized_model_RF.sav'
filename2 = 'finalized_model_GBC.sav'
filename3 = 'finalized_model_SVC.sav'
filename4 = 'model_acc_914_auc_974.h5'

rf_model = joblib.load(filename1)
gb_model = joblib.load(filename2)
sv_model = joblib.load(filename3)
cn_model = load_model(filename4)
print("Loaded model from disk")

# mengambil nilai tengah
X_train_rf = X_train[:,2,2,:]
X_test_rf = X_test[:,2,2,:]

result1 = rf_model.score(X_test_rf, y_test)
result2 = gb_model.score(X_test_rf, y_test)
result3 = sv_model.score(X_test_rf, y_test)
result4 = cn_model.evaluate(X_test, y_test)
print(result1)
print(result2)
print(result3)
print("[test loss, test accuracy]:", result4)

rf_probs = rf_model.predict_proba(X_test_rf)
gb_probs = gb_model.predict_proba(X_test_rf)
sv_probs = sv_model.predict_proba(X_test_rf)
cn_probs = cn_model.predict(X_test).ravel()


#%%
Demonstrating the calculation of this significance test.

# Wilcoxon signed-rank test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import wilcoxon
from numpy import load
import numpy as np
import os
# seed the random number generator
seed(1)
# generate two independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51

os.chdir('G:\\My Drive\\DATA')

cn_probs = np.load('prediksi25.npy')
sv_probs = np.load('prediksi_sv.npy')
gb_probs = np.load('prediksi_gb.npy')
rf_probs = np.load('prediksi_rf.npy')
nn_probs = np.load('prediksi_nn.npy')

cn_det = [round(num) for num in cn_probs]
sv_det = [round(num) for num in sv_probs]
gb_det = [round(num) for num in gb_probs]
rf_det = [round(num) for num in rf_probs]
nn_det = [round(num) for num in nn_probs]

# compare samples
print('===========================CNN vs SVC===================================')
stat, p_sv = wilcoxon(cn_det, sv_det[:60258])
print('Statistics=%.5f, p=%.5f' % (stat, p_sv))
# interpret
alpha = 0.05

if p > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')
    
# compare samples
print('===========================CNN vs GBC===================================')
stat, p_gb = wilcoxon(cn_det, gb_det[:60258])
print('Statistics=%.5f, p=%.5f' % (stat, p_gb))
# interpret
alpha = 0.05

if p > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')
    
# compare samples
print('===========================CNN vs RF===================================')
stat, p_rf = wilcoxon(cn_det, rf_det[:60258])
print('Statistics=%.5f, p=%.5f' % (stat, p_rf))
# interpret
alpha = 0.05

if p > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')
    
# compare samples
print('===========================SVC vs RF===================================')
stat, p_nn = wilcoxon(cn_det, nn_det[:60258])
print('Statistics=%.5f, p=%.5f' % (stat, p_nn))
# interpret
alpha = 0.05

if p > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')
    
# compare samples
print('===========================GBC vs RF===================================')
stat, p = wilcoxon(gb_probs[:,1], rf_probs[:,1])
print('Statistics=%.5f, p=%.5f' % (stat, p))
# interpret
alpha = 0.05

if p > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')
    
# compare samples
print('===========================GBC vs SVC===================================')
stat, p = wilcoxon(gb_probs[:,1], sv_probs[:,1])
print('Statistics=%.5f, p=%.5f' % (stat, p))
# interpret
alpha = 0.05

if p > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')
  

    
    
    
    
