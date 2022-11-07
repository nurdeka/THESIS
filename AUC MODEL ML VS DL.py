# MODEL ML VS DL
#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
import os

#%%
os.chdir('C:\\Users\\nurde\\DATA_THESIS\\HS_BARU_20220616')
os.getcwd()

df = pd.read_csv('train_pertitik.csv')
df1 = pd.read_csv('test_pertitik.csv')

df.dropna(inplace=True)

df1.dropna(inplace=True)
#df.reset_index(drop=True, inplace=True)

print(df.isna().sum())

print(df1.isna().sum())

print(df.shape)
print(df1.shape)

df_x = df.iloc[:,5:]
print('df_x shape is :', df_x.shape)
df_x1 = df1.iloc[:,5:]
print('df_x shape is :', df_x1.shape)

df_y = df.iloc[:,4]
df_y=df_y.astype('int')
print('df_y shape is :', df_y.shape)
df_y1 = df1.iloc[:,4]
print('df_y shape is :', df_y1.shape)

arr_x = df_x.values
arr_y = df_y.values
arr_x1 = df_x1.values
arr_y1 = df_y1.values

X_trains = arr_x
y_trains = arr_y
X_tests = arr_x1
y_tests = arr_y1

print(X_trains.shape)
print(X_tests.shape)
print(y_trains.shape)
print(y_tests.shape)

from numpy import load
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

#X_trains, X_val, y_trains, y_val = train_test_split(X_train, y_train, random_state=1, stratify=y)


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
model1.fit(X_trains, y_trains)
model2.fit(X_trains, y_trains)
model3.fit(X_trains, y_trains)

#%%
# evaluate the model
#cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
#n_scores1 = cross_val_score(model1, X_trains, y_trains, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
#n_scores2 = cross_val_score(model2, X_trains, y_trains, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
#n_scores3 = cross_val_score(model3, X_trains, y_trains, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
#print('Accuracy RF: %.3f (%.3f)' % (mean(n_scores1), std(n_scores1)))
#print('Accuracy GBC: %.3f (%.3f)' % (mean(n_scores2), std(n_scores2)))
#print('Accuracy SVC: %.3f (%.3f)' % (mean(n_scores3), std(n_scores3)))
#%%
# Make predictions for the test set
y_pred_test1 = model1.predict(X_tests)
y_pred_test2 = model2.predict(X_tests)
y_pred_test3 = model3.predict(X_tests)
# View accuracy score
print('RF accuracy : ', (metrics.accuracy_score(y_tests, y_pred_test1)))
print('GBC accuracy : ', (metrics.accuracy_score(y_tests, y_pred_test2)))
print('SVC accuracy : ', (metrics.accuracy_score(y_tests, y_pred_test3)))

#%%
import joblib
import tensorflow
from tensorflow.keras.models import load_model
# save the model to disk
filename1 = 'finalized_model_RF.sav'
filename2 = 'finalized_model_GBC.sav'
filename3 = 'finalized_model_SVC.sav'
filename4 = 'model_acc_914_auc_974.h5'
joblib.dump(model1, filename1)
joblib.dump(model2, filename2)
joblib.dump(model3, filename3)
#loaded_model.load_weights(filename4)

 
# some time later...
 
# load the model from disk

rf_model = joblib.load(filename1)
gb_model = joblib.load(filename2)
sv_model = joblib.load(filename3)
cn_model = load_model(filename4)
print("Loaded model from disk")



result1 = rf_model.score(X_tests, y_tests)
result2 = gb_model.score(X_tests, y_tests)
result3 = sv_model.score(X_tests, y_tests)
result4 = cn_model.evaluate(X_test, y_test)
print(result1)
print(result2)
print(result3)
print("[test loss, test accuracy]:", result4)

#%% ROC AUC
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
ns_probs = [0 for _ in range(len(y_tests))]
# predict probabilities
rf_probs = rf_model.predict_proba(X_tests)
gb_probs = gb_model.predict_proba(X_tests)
sv_probs = sv_model.predict_proba(X_tests)
cn_probs = cn_model.predict(X_test).ravel()
#gb_probs = model5.predict_proba(X_test)
#ab_probs = model6.predict_proba(X_test)
# keep probabilities for the positive outcome only
rf_probs = rf_probs[:, 1]
gb_probs = gb_probs[:, 1]
sv_probs = sv_probs[:, 1]
#xg_probs = xg_probs[:, 1]
#gb_probs = gb_probs[:, 1]
#ab_probs = ab_probs[:, 1]
# calculate scores

from sklearn.metrics import roc_auc_score
ns_auc = roc_auc_score(y_tests, ns_probs)
rf_auc = roc_auc_score(y_tests, rf_probs)
gb_auc = roc_auc_score(y_tests, gb_probs)
sv_auc = roc_auc_score(y_tests, sv_probs)
#xg_auc = roc_auc_score(y_test, xg_probs)
#gb_auc = roc_auc_score(y_test, gb_probs)
#ab_auc = roc_auc_score(y_test, ab_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Random Forest: ROC AUC=%.3f' % (rf_auc))
print('Gradien Boost: ROC AUC=%.3f' % (gb_auc))
print('Support Vector: ROC AUC=%.3f' % (sv_auc))
#print('XGB: ROC AUC=%.3f' % (xg_auc))
#print('GBC: ROC AUC=%.3f' % (gb_auc))
#print('ABC: ROC AUC=%.3f' % (ab_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_tests, ns_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_tests, rf_probs)
gb_fpr, gb_tpr, _ = roc_curve(y_tests, gb_probs)
sv_fpr, sv_tpr, _ = roc_curve(y_tests, sv_probs)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, cn_probs)
#xg_fpr, xg_tpr, _ = roc_curve(y_test, xg_probs)
#gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_probs)
#ab_fpr, ab_tpr, _ = roc_curve(y_test, ab_probs)
auc_keras = auc(fpr_keras, tpr_keras)
print('CNN Keras: ROC AUC=%.3f' % (auc_keras))
# plot the roc curve for the model
pyplot.figure(figsize=(8, 8))
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(sv_fpr, sv_tpr, label='SVC (AUC = {:.3f})'.format(sv_auc))
pyplot.plot(gb_fpr, gb_tpr, label='GBC (AUC= {:.3f})'.format(gb_auc))
pyplot.plot(rf_fpr, rf_tpr, label='RF (AUC = {:.3f})'.format(rf_auc))
pyplot.plot(fpr_keras, tpr_keras, label='Proposed CNN (AUC = {:.3f})'.format(auc_keras))


#pyplot.plot(kn_fpr, kn_tpr, marker='.', label='KNN')
#pyplot.plot(xg_fpr, xg_tpr, marker='.', label='XGB')
#pyplot.plot(rf_fpr, rf_tpr, marker='.', label='RF')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.title('ROC curve')
# show the legend
pyplot.legend(loc='best')
# show the plot
pyplot.show()


# make a prediction
cn_probab = cn_model.predict_proba(X_test)

#################################
import scipy.stats as stats
group_CNN = pred list[]
group_RF = pred list[]
group_GBC = pred list[]
group_SVC = pred list[]
group_NN = pred list[]
#perform the Wilcoxon-Signed Rank Test
print('CNN vs RF :', stats.wilcoxon(group_CNN, group_RF))
print('CNN vs GBC :', stats.wilcoxon(group_CNN, group_GBC))
print('CNN vs SVC :', stats.wilcoxon(group_CNN, group_SVC))
print('CNN vs NN :', stats.wilcoxon(group_CNN, group_NN))