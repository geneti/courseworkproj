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

train_ordinal_copy = (train_ordinal - train_ordinal.min())/(train_ordinal.max() - train_ordinal.min())
test_ordinal_copy = (test_ordinal - test_ordinal.min())/(test_ordinal.max() - test_ordinal.min())

#train_ordinal = (train_ordinal - train_ordinal.min())/(train_ordinal.std())
#test_ordinal = (test_ordinal - test_ordinal.min())/(test_ordinal.std())

#train_ordinal = normalize(train_ordinal, norm = 'l1', axis = 0)
#test_ordinal = normalize(test_ordinal, norm = 'l1', axis = 0)

#train_ordinal = normalize(train_ordinal, norm = 'l2', axis = 0)
#test_ordinal = normalize(test_ordinal, norm = 'l2', axis = 0)

# feature reduction

nc = 10
pca1 = PCA(n_components=nc, svd_solver='full')
train_ordinal_copy = pca1.fit_transform(train_ordinal_copy)
pca2 = PCA(n_components=nc, svd_solver='full')
test_ordinal_copy = pca2.fit_transform(test_ordinal_copy)

# transform to pandas dataframe
train_ordinal_copy = pd.DataFrame(train_ordinal_copy)
test_ordinal_copy = pd.DataFrame(test_ordinal_copy)

R = range(1,37)
acc_dr = []
acc_or = []
f, ax = plt.subplots(1)

for k in R:
    model1 = KNeighborsClassifier(n_neighbors=k)
    model1.fit(train_ordinal, train_label)
    pred1 = model1.predict(test_ordinal)
    acc_or.append(accuracy_score(test_label, pred1))

    model2 = KNeighborsClassifier(n_neighbors=k)
    model2.fit(train_ordinal_copy, train_label)
    pred2 = model2.predict(test_ordinal_copy)
    acc_dr.append(accuracy_score(test_label, pred2))

ax.plot(R, acc_or, label="KNN without Norm and PCA", alpha=0.7)
ax.plot(R, acc_dr, label="KNN with Norm and PCA", alpha=0.7)

bayes_base = [0.6552]*len(R)
ax.plot(R, bayes_base, label="Naive Bayesian with ordinal features", alpha=0.7,color='r')

plt.legend()
plt.show()