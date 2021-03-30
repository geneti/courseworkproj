import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import seaborn as sn
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
from DataLoad import dataload
from Classifier.Bayes.NaiveBayes import NaiveBayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.svm import SVC
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# Define the model
model = SVC()

# load data
train = dataload('./train.csv')
train_data = train.get_data()
train_ordinal = train.get_ordinal_data()
train_nominal = train.get_nominal_data()
missing_ordinal = train.get_ordinal_mean()
train_label = train.get_label()

test = dataload('./test.csv', missing_ordinal)
test_data = test.get_data()
test_ordinal = test.get_ordinal_data()
test_nominal = test.get_nominal_data()
test_label = test.get_label()

# normalization

train_ordinal = (train_ordinal - train_ordinal.min())/(train_ordinal.max() - train_ordinal.min())
test_ordinal = (test_ordinal - test_ordinal.min())/(test_ordinal.max() - test_ordinal.min())

#train_ordinal = (train_ordinal - train_ordinal.min())/(train_ordinal.std())
#test_ordinal = (test_ordinal - test_ordinal.min())/(test_ordinal.std())

#train_ordinal = normalize(train_ordinal, norm = 'l1', axis = 0)
#test_ordinal = normalize(test_ordinal, norm = 'l1', axis = 0)

#train_ordinal = normalize(train_ordinal, norm = 'l2', axis = 0)
#test_ordinal = normalize(test_ordinal, norm = 'l2', axis = 0)

# feature reduction

sfs = SFS(model,k_features=10, forward=True, floating=False, verbose=2,scoring='accuracy',cv=2)
sbs = SFS(model,k_features=10, forward=False, floating=False, verbose=2,scoring='accuracy',cv=2)
sffs = SFS(model,k_features=10, forward=True, floating=True, verbose=2,scoring='accuracy',cv=2)
sbfs = SFS(model,k_features=10, forward=False, floating=True, verbose=2,scoring='accuracy',cv=2)

# train and test model

Model = sbfs

'''
scores = cross_val_score(Model, train_ordinal, train_label, cv=5)
print(scores)
print("Score Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
'''

Model.fit(train_ordinal, train_label)
print('CV Score:', Model.k_feature_names_)
print(type(Model.k_feature_names_[0]))

# dimension reduction

train_ordinal = train_ordinal.loc[:,Model.k_feature_names_]
test_ordinal = test_ordinal.loc[:,Model.k_feature_names_]

model.fit(train_ordinal, train_label)

pred = model.predict(test_ordinal)
pd.set_option('precision', 4)

print('The accuracy is: %0.4f'% accuracy_score(test_label, pred))
classes = np.sort(pd.unique(train_label))
cm = pd.DataFrame(confusion_matrix(test_label, pred), index = classes, columns = classes)
print('labels accuracy is: \n', np.diag(cm) / cm.sum())
'''
sn.heatmap(cm, annot=True, linewidths=1, cmap="YlOrRd")
plt.xlabel('Predict')
plt.ylabel('True')
plt.show()
'''