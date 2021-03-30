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

f, ax = plt.subplots(1)

pca_list = []
sfs_list = []
sffs_list = []

R = range(1,37)

for k in R:
    # feature reduction
    pca1 = PCA(n_components=k)
    train_ordinal_pca = pca1.fit_transform(train_ordinal)
    pca2 = PCA(n_components=k)
    test_ordinal_pca = pca2.fit_transform(test_ordinal)

    sfs = SFS(model,k_features=k, forward=True, floating=False, verbose=2,scoring='accuracy',cv=2)
    #sbs = SFS(model,k_features=k, forward=False, floating=False, verbose=2,scoring='accuracy',cv=2)
    sffs = SFS(model,k_features=k, forward=True, floating=True, verbose=2,scoring='accuracy',cv=2)
    #sbfs = SFS(model,k_features=k, forward=False, floating=True, verbose=2,scoring='accuracy',cv=2)

    sfs.fit(train_ordinal, train_label)
    train_ordinal_sfs = train_ordinal.loc[:,sfs.k_feature_names_]
    test_ordinal_sfs = test_ordinal.loc[:,sfs.k_feature_names_]

    sffs.fit(train_ordinal, train_label)
    train_ordinal_sffs = train_ordinal.loc[:,sffs.k_feature_names_]
    test_ordinal_sffs = test_ordinal.loc[:,sffs.k_feature_names_]

    # pca accuracy
    model.fit(train_ordinal_pca, train_label)
    pred_pca = model.predict(test_ordinal_pca)
    acc_pca = accuracy_score(test_label, pred_pca)

    # sfs accuracy
    model.fit(train_ordinal_sfs, train_label)
    pred_sfs = model.predict(test_ordinal_sfs)
    acc_sfs = accuracy_score(test_label, pred_sfs)

    # sffs accuracy
    model.fit(train_ordinal_sffs, train_label)
    pred_sffs = model.predict(test_ordinal_sffs)
    acc_sffs = accuracy_score(test_label, pred_sffs)

    pca_list.append(acc_pca)
    sfs_list.append(acc_sfs)
    sffs_list.append(acc_sffs)

ax.plot(R, pca_list, label="PCA", alpha=0.7)
ax.plot(R, sfs_list, label="SFS", alpha=0.7)
ax.plot(R, sffs_list, label="SFFS", alpha=0.7)

plt.legend()
plt.show()
print('\n')
print('pca',pca_list)
print('sfs',sfs_list)
print('sffs',sffs_list)
