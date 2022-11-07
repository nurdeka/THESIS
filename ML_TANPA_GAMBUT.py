# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 00:47:36 2022

@author: nurde
"""

from numpy import load
import os

os.chdir('C:\\Users\\nurde\\DATA_THESIS\HS_BARU_20220616')
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
X_train= X_train[:,2,2,:]
X_test = X_test[:,2,2,:]

import numpy as np
# Eksperimen Tanpa Gambut
X_trains = np.stack((X_train[:,0], X_train[:,1], X_train[:,2], X_train[:,3], X_train[:,4],
                     X_train[:,5], X_train[:,6], X_train[:,8], X_train[:,9], X_train[:,10]), axis=-1)
X_tests = np.stack((X_test[:,0], X_test[:,1], X_test[:,2], X_test[:,3], X_test[:,4],
                     X_test[:,5], X_test[:,6], X_test[:,8], X_test[:,9], X_test[:,10]), axis=-1)
print(X_trains.shape)
print(X_tests.shape)



#%%
# %% ML
import numpy
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn import metrics


# define the model
model1 = RandomForestClassifier()
model2 = GradientBoostingClassifier()
model3 = SVC(probability=True)

# fit the model
model1.fit(X_trains, y_train)
model2.fit(X_trains, y_train)
print('GBC OK')
model3.fit(X_trains, y_train)
print('SVC OK')


result1 = model1.score(X_tests, y_test)
result2 = model2.score(X_tests, y_test)
result3 = model3.score(X_tests, y_test)

print('score RF', result1)
print('score GBC', result2)
print('score SVC', result3)

#%% ROC AUC
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
rf_model = model1
gb_model = model2
rf_probs = rf_model.predict_proba(X_tests)
gb_probs = gb_model.predict_proba(X_tests)
sv_probs = sv_model.predict_proba(X_tests)
cn_probs = cn_model.predict(X_test).ravel()
nn_probs = model.predict_proba(X_test)
#ab_probs = model6.predict_proba(X_test)
# keep probabilities for the positive outcome only
rf_probs = rf_probs[:, 1]
gb_probs = gb_probs[:, 1]
sv_probs = sv_probs[:, 1]
nn_probs = nn_probs.ravel()
#gb_probs = gb_probs[:, 1]
#ab_probs = ab_probs[:, 1]
# calculate scores

from sklearn.metrics import roc_auc_score
ns_auc = roc_auc_score(y_test, ns_probs)
rf_auc = roc_auc_score(y_test, rf_probs)
gb_auc = roc_auc_score(y_test, gb_probs)
sv_auc = roc_auc_score(y_test, sv_probs)
nn_auc = roc_auc_score(y_test, nn_probs)
#gb_auc = roc_auc_score(y_test, gb_probs)
#ab_auc = roc_auc_score(y_test, ab_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Random Forest: ROC AUC=%.3f' % (rf_auc))
print('Gradien Boost: ROC AUC=%.3f' % (gb_auc))
print('Support Vector: ROC AUC=%.3f' % (sv_auc))
print('Neural Network: ROC AUC=%.3f' % (nn_auc))
#print('GBC: ROC AUC=%.3f' % (gb_auc))
#print('ABC: ROC AUC=%.3f' % (ab_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_probs)
sv_fpr, sv_tpr, _ = roc_curve(y_test, sv_probs)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, cn_probs)
nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_probs)
#gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_probs)
#ab_fpr, ab_tpr, _ = roc_curve(y_test, ab_probs)
auc_keras = auc(fpr_keras, tpr_keras)
print('CNN Keras: ROC AUC=%.3f' % (auc_keras))

# plot the roc curve for the model
plt.figure(figsize=(7, 7))
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')

#plt.plot(sv_fpr, sv_tpr, label='SVC (AUC = {:.3f})'.format(sv_auc))
plt.plot(gb_fpr, gb_tpr, label='GBC (AUC= {:.3f})'.format(gb_auc))
#plt.plot(nn_fpr, nn_tpr, label='NN (AUC = {:.3f})'.format(nn_auc))
plt.plot(rf_fpr, rf_tpr, label='RF (AUC = {:.3f})'.format(rf_auc))
#plt.plot(fpr_keras, tpr_keras, label='Proposed CNN (AUC = {:.3f})'.format(auc_keras))


#pyplot.plot(kn_fpr, kn_tpr, marker='.', label='KNN')
#pyplot.plot(xg_fpr, xg_tpr, marker='.', label='XGB')
#pyplot.plot(rf_fpr, rf_tpr, marker='.', label='RF')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
# show the legend
plt.legend(loc='best')
# show the plot
plt.show()

filename1 = 'finalized_model_RF_nopeat.sav'
filename2 = 'finalized_model_GBC_nopeat.sav'
#filename3 = 'finalized_model_SVC.sav'
#filename4 = 'model_acc_914_auc_974.h5'
joblib.dump(model1, filename1)
joblib.dump(model2, filename2)

