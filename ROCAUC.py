%%time
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
# load the model from disk
model1 = joblib.load('finalized_model_RF1.sav')
model2 = joblib.load('finalized_model_DT1.sav')
model3 = joblib.load('finalized_model_KNN3.sav')
model4 = joblib.load('finalized_model_XGB.sav')
model5 = joblib.load('finalized_model_GBC.sav')
model6 = joblib.load('finalized_model_ABC.sav')

#predictions = loaded_model.predict(X_test)
#predictions_proba = loaded_model.predict_proba(X_test)
#acc = accuracy_score(y_test, predictions)
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
rf_probs = model1.predict_proba(X_test)
dt_probs = model2.predict_proba(X_test)
kn_probs = model3.predict_proba(X_test)
xg_probs = model4.predict_proba(X_test)
gb_probs = model5.predict_proba(X_test)
ab_probs = model6.predict_proba(X_test)
# keep probabilities for the positive outcome only
rf_probs = rf_probs[:, 1]
dt_probs = dt_probs[:, 1]
kn_probs = kn_probs[:, 1]
xg_probs = xg_probs[:, 1]
gb_probs = gb_probs[:, 1]
ab_probs = ab_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
rf_auc = roc_auc_score(y_test, rf_probs)
dt_auc = roc_auc_score(y_test, dt_probs)
kn_auc = roc_auc_score(y_test, kn_probs)
xg_auc = roc_auc_score(y_test, xg_probs)
gb_auc = roc_auc_score(y_test, gb_probs)
ab_auc = roc_auc_score(y_test, ab_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Random Forest: ROC AUC=%.3f' % (rf_auc))
print('Decision Tree: ROC AUC=%.3f' % (dt_auc))
print('KNear Neighbour: ROC AUC=%.3f' % (kn_auc))
print('XGB: ROC AUC=%.3f' % (xg_auc))
print('GBC: ROC AUC=%.3f' % (gb_auc))
print('ABC: ROC AUC=%.3f' % (ab_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs)
kn_fpr, kn_tpr, _ = roc_curve(y_test, kn_probs)
xg_fpr, xg_tpr, _ = roc_curve(y_test, xg_probs)
gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_probs)
ab_fpr, ab_tpr, _ = roc_curve(y_test, ab_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='DT')
pyplot.plot(gb_fpr, gb_tpr, marker='.', label='GBC')
pyplot.plot(ab_fpr, ab_tpr, marker='.', label='ABC')
pyplot.plot(kn_fpr, kn_tpr, marker='.', label='KNN')
pyplot.plot(xg_fpr, xg_tpr, marker='.', label='XGB')
pyplot.plot(rf_fpr, rf_tpr, marker='.', label='RF')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()