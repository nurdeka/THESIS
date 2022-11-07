# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:20:29 2022

@author: nurde
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot

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

#%% TRAIN TEST SPLIT
X_trains = arr_x
y_trains = arr_y
X_tests = arr_x1
y_tests = arr_y1

#%%# random forest for feature importance on a classification problem
# define dataset
#X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define the model
model = RandomForestClassifier()
# fit the model
model.fit(arr_x, arr_y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)

pyplot.show()

#%%AM VALUE
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = df_x.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(df_x.values, i)
                          for i in range(len(df_x.columns))]
  
print(vif_data)



#%%time
# evaluate random forest algorithm for classification
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


# define the model
model = RandomForestClassifier(n_estimators=50, 
                               bootstrap = True,
                               max_features = 'sqrt')
model = GradientBoostingClassifier()

model = SVC(probability=True)

# fit the model
model.fit(x_train, y_train)
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
n_scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# Make predictions for the test set
y_pred_test = model.predict(x_test)

# View accuracy score
accuracy_score(y_test, y_pred_test)

#%% RANDOM FOREST
# Import needed packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Instantiate and fit the RandomForestClassifier
forest = RandomForestClassifier()
forest.fit(x_train, y_train)

# Make predictions for the test set
y_pred_test = forest.predict(x_test)

# View accuracy score
accuracy_score(y_test, y_pred_test)

# View confusion matrix for test data and predictions
confusion_matrix(y_test, y_pred_test)

#%%# SAVE AND LOAD MODEL
import joblib
# save the model to disk
filename = 'finalized_model_GBC_10fold.sav'
joblib.dump(model, filename)
 
# some time later...
 
# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(x_test, y_test)
print(result)



#%%# ALL
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(arr_x, arr_y, test_size=0.2)
# importing package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
# evaluate random forest algorithm for classification
import numpy
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

# mendefinisikan classifiers
classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
	AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
    XGBClassifier()]

log_cols = ["Classifier", "Accuracy"]
log 	 = pd.DataFrame(columns=log_cols)

#sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

#X = df.iloc[:, 1:]
#y = df.iloc[:, 0]

acc_dict = {}

#%%time
# Running models
for clf in classifiers:
	name = clf.__class__.__name__
	clf.fit(X_train, y_train)
	train_predictions = clf.predict(X_test)
	acc = accuracy_score(y_test, train_predictions)
	if name in acc_dict:
		acc_dict[name] += acc
	else:
		acc_dict[name] = acc

for clf in acc_dict:
	acc_dict[clf] = acc_dict[clf]
	log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
	log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
print(log)


##IGR Methode
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
model = RandomForestClassifier(criterion='entropy')
# fit the model
model.fit(arr_x, arr_y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


