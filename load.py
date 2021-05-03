import pandas as pd
import time
from sklearn import svm 
from sklearn.model_selection import cross_validate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from yellowbrick.model_selection import CVScores
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.classifier import confusion_matrix
from sklearn.model_selection import train_test_split as tts

start_time = time.time();
SCORING = ["f1_macro", "precision", "accuracy", "recall", "roc_auc"]
results = pd.read_pickle("/scratch2/2191051t/results.pkl")

elp_time = time.time() - start_time

print(elp_time)
results.head()
row = results.iloc[0]

results = results[results.newState != 'canceled']
results = results[np.isfinite(results.lenLogs)]
#langResults = {} 
#langResults['python'] = results[results['language'] == 'python']
#langResults['node'] = results[results['language'] == 'node.js']
#langResults['java'] = results[results['language'] == 'java']
#langResults['ruby'] = results[results['language'] == 'ruby']

resultsState = results['newState']
#langResultsState = {}
#langResultsState['python'] = langResults['python']['newState']
#langResultsState['node'] = langResults['node']['newState']
#langResultsState['java'] = langResults['java']['newState']
#langResultsState['ruby'] = langResults['ruby']['newState']

resultsState[results.newState == 'errored'] = 'failed' 
#langResultsState['python'][langResults['python']['newState'] == 'errored'] = 'failed' 
#langResultsState['node'][langResults['node']['newState'] == 'errored'] = 'failed' 
#langResultsState['java'][langResults['java']['newState'] == 'errored'] = 'failed' 
#langResultsState['ruby'][langResults['ruby']['newState'] == 'errored'] = 'failed' 

print(results.shape)
print(resultsState.shape)
del results['new']
del results['newState']
del results['branch']
del results['language']
del results['repoLanguage']
del results['log']
del results['repository_slug']
del results['commit']
del results['id']
del results['_id']
del results['_id_y']
#for results in langResults:
#	del langResults[results]['new']
#        del langResults[results]['newState']
#	del langResults[results]['branch']
#	del langResults[results]['language']
#	del langResults[results]['repoLanguage']
#	del langResults[results]['log']
#	del langResults[results]['repository_slug']
#	del langResults[results]['commit']
#	del langResults[results]['id']
#	del langResults[results]['_id']
#	del langResults[results]['_id_y']
#	print(langResults[results].columns)
resultsState.fillna('0', inplace=True)
resultsState = resultsState.replace(['failed', 'passed'], [0, 1])

x_train, x_test, y_train, y_test = tts(results, resultsState, test_size=0.2)

clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000) 
sgdscores = cross_validate(clf, results, resultsState, cv=5, scoring=SCORING)
vis = FeatureImportances(clf)
vis.fit(results, resultsState)
vis.show(outpath="FI1.png")
plt.clf()

cm = confusion_matrix(clf, x_train, y_train, x_test, y_test, classes=['Failed rerun', 'Successful rerun'])
cm.show(outpath="CM1.png")
plt.clf()

clf = svm.LinearSVC(loss="hinge", max_iter=5000)
vis = FeatureImportances(clf)
vis.fit(results, resultsState)
vis.show(outpath="FI2.png")
plt.clf()
cm = confusion_matrix(clf, x_train, y_train, x_test, y_test, classes=['Failed rerun', 'Successful rerun'])
cm.show(outpath="CM2.png")
plt.clf()
svcscores = cross_validate(clf, results, resultsState, cv=5, scoring=SCORING)

clf = DummyClassifier(strategy="most_frequent")

cm = confusion_matrix(clf, x_train, y_train, x_test, y_test, classes=['Failed rerun', 'Successful rerun'])
cm.show(outpath="CM3.png")
plt.clf()
dumbscores = cross_validate(clf, results, resultsState, cv=5, scoring=SCORING)

clf = GaussianNB()
nbscores = cross_validate(clf, results, resultsState, cv=5, scoring=SCORING)
#vis = FeatureImportances(clf)
#vis.fit(results, resultsState)
#vis.show(outpath="FI4.png")

cm = confusion_matrix(clf, x_train, y_train, x_test, y_test, classes=['Failed rerun', 'Successful rerun'])
cm.show(outpath="CM4.png")
plt.clf()
clf = RandomForestClassifier(n_estimators=10, random_state=1)
rfscores = cross_validate(clf, results, resultsState, cv=5, scoring=SCORING)

vis = FeatureImportances(clf, relative=False)
vis.fit(results, resultsState)
vis.show(outpath="FI5.png")
plt.clf()
cm = confusion_matrix(clf, x_train, y_train, x_test, y_test, classes=['Failed rerun', 'Successful rerun'])
cm.show(outpath="CM5.png")
plt.clf()
cv = StratifiedKFold(n_splits=12, random_state=42)

visualizer = CVScores(clf, cv=cv, scoring='f1_weighted')

visualizer.fit(results, resultsState)
visualizer.show(outpath="CV5.png")

labels = ['SGD', 'SVC', 'Most Frequent', 'Naive Bayes', 'Random Forest']
width = 1
x = np.arange(len(labels))
fig, ax = plt.subplots()
rects1 = ax.bar(0, sgdscores['test_f1_macro'].mean(), width, zorder=1)
rects1 = ax.bar(1, svcscores['test_f1_macro'].mean(), width, zorder=1)
rects1 = ax.bar(2, dumbscores['test_f1_macro'].mean(), width, zorder=1)
rects1 = ax.bar(3, nbscores['test_f1_macro'].mean(), width, zorder=1)
rects1 = ax.bar(4, rfscores['test_f1_macro'].mean(), width, zorder=1)
ax.set_ylabel('F1 Scores')
ax.set_title('Comparing F1 scores between classifiers')
ax.set_xticks(x)
ax.set_xticklabels(labels)
for i in range(5):
    ax.scatter(0, sgdscores['test_f1_macro'][i], color='black', zorder=5)
    ax.scatter(1, svcscores['test_f1_macro'][i], color='black', zorder=5)
    ax.scatter(2, dumbscores['test_f1_macro'][i], color='black', zorder=5)
    ax.scatter(3, nbscores['test_f1_macro'][i], color='black', zorder=5)
    ax.scatter(4, rfscores['test_f1_macro'][i], color='black', zorder=5)
fig.tight_layout()
plt.savefig('Fig1.png')

fig, ax = plt.subplots()
rects1 = ax.bar(0, sgdscores['test_precision'].mean(), width)
rects1 = ax.bar(1, svcscores['test_precision'].mean(), width)
rects1 = ax.bar(2, dumbscores['test_precision'].mean(), width)
rects1 = ax.bar(3, nbscores['test_precision'].mean(), width)
rects1 = ax.bar(4, rfscores['test_precision'].mean(), width)
ax.set_ylabel('Precision')
ax.set_title('Comparing Precision between classifiers')
ax.set_xticks(x)
ax.set_xticklabels(labels)
for i in range(5):
    ax.scatter(0, sgdscores['test_precision'][i], color='black', zorder=5)
    ax.scatter(1, svcscores['test_precision'][i], color='black', zorder=5)
    ax.scatter(2, dumbscores['test_precision'][i], color='black', zorder=5)
    ax.scatter(3, nbscores['test_precision'][i], color='black', zorder=5)
    ax.scatter(4, rfscores['test_precision'][i], color='black', zorder=5)
fig.tight_layout()
plt.savefig('Fig2.png')

fig, ax = plt.subplots()
rects1 = ax.bar(0, sgdscores['test_accuracy'].mean(), width)
rects1 = ax.bar(1, svcscores['test_accuracy'].mean(), width)
rects1 = ax.bar(2, dumbscores['test_accuracy'].mean(), width)
rects1 = ax.bar(3, nbscores['test_accuracy'].mean(), width)
rects1 = ax.bar(4, rfscores['test_accuracy'].mean(), width)
ax.set_ylabel('Accuracy')
ax.set_title('Comparing Accuracy between classifiers')
ax.set_xticks(x)
ax.set_xticklabels(labels)
for i in range(5):
    ax.scatter(0, sgdscores['test_accuracy'][i], color='black', zorder=5)
    ax.scatter(1, svcscores['test_accuracy'][i], color='black', zorder=5)
    ax.scatter(2, dumbscores['test_accuracy'][i], color='black', zorder=5)
    ax.scatter(3, nbscores['test_accuracy'][i], color='black', zorder=5)
    ax.scatter(4, rfscores['test_accuracy'][i], color='black', zorder=5)

fig.tight_layout()
plt.savefig('Fig3.png')


fig, ax = plt.subplots()
rects1 = ax.bar(0, sgdscores['test_recall'].mean(), width)
rects1 = ax.bar(1, svcscores['test_recall'].mean(), width)
rects1 = ax.bar(2, dumbscores['test_recall'].mean(), width)
rects1 = ax.bar(3, nbscores['test_recall'].mean(), width)
rects1 = ax.bar(4, rfscores['test_recall'].mean(), width)
ax.set_ylabel('Recall')
ax.set_title('Comparing Recall between classifiers')
ax.set_xticks(x)
ax.set_xticklabels(labels)
for i in range(5):
    ax.scatter(0, sgdscores['test_recall'][i], color='black', zorder=5)
    ax.scatter(1, svcscores['test_recall'][i], color='black', zorder=5)
    ax.scatter(2, dumbscores['test_recall'][i], color='black', zorder=5)
    ax.scatter(3, nbscores['test_recall'][i], color='black', zorder=5)
    ax.scatter(4, rfscores['test_recall'][i], color='black', zorder=5)

fig.tight_layout()
plt.savefig('Fig4.png')

fig, ax = plt.subplots()
rects1 = ax.bar(0, sgdscores['test_roc_auc'].mean(), width)
rects1 = ax.bar(1, svcscores['test_roc_auc'].mean(), width)
rects1 = ax.bar(2, dumbscores['test_roc_auc'].mean(), width)
rects1 = ax.bar(3, nbscores['test_roc_auc'].mean(), width)
rects1 = ax.bar(4, rfscores['test_roc_auc'].mean(), width)
ax.set_ylabel('ROC AUC')
ax.set_title('Comparing roc_auc between classifiers')
ax.set_xticks(x)
ax.set_xticklabels(labels)
for i in range(5):
    ax.scatter(0, sgdscores['test_roc_auc'][i], color='black', zorder=5)
    ax.scatter(1, svcscores['test_roc_auc'][i], color='black', zorder=5)
    ax.scatter(2, dumbscores['test_roc_auc'][i], color='black', zorder=5)
    ax.scatter(3, nbscores['test_roc_auc'][i], color='black', zorder=5)
    ax.scatter(4, rfscores['test_roc_auc'][i], color='black', zorder=5)

fig.tight_layout()
plt.savefig('Fig5.png')



print(str(time.time() - start_time) + 's')




