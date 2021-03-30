import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import operator
from sklearn.metrics import accuracy_score

# Designed for discrete features with string type
'''
eps: the laplace smoothing parameter
labels: pandas Series
data: pandas DataFrame
log(postrior) = log(prior) + log(likelihood) - log(evidence)
'''
class NaiveBayes(object):
	def __init__(self):
		self.dimension = 0
		self.labels = pd.DataFrame()
		self.likelihood = dict()
		self.prior = dict()
		self.eps = 0
		self.classes = pd.DataFrame()
		
	def fit(self, X, Y, eps = 0.01):
		# Get basic info of labels
		self.labels = Y
		self.classes = np.sort(pd.unique(self.labels))
		self.likelihood = dict.fromkeys(self.classes)
		for item in self.likelihood.keys():
			self.likelihood[item] = dict.fromkeys(range(X.shape[1]))
			for key in self.likelihood[item].keys():
				self.likelihood[item][key] = dict()
		self.prior = dict.fromkeys(self.classes, 0)
		self.unknown = dict.fromkeys(self.classes)
		for key in self.unknown:
			self.unknown[key] = dict()
		self.eps = eps

		for i in range(X.shape[0]):
			current_label = self.labels[i]
			self.prior[current_label] += 1
			for j in range(X.shape[1]):
				val = X.iloc[i,j]
				if val not in self.likelihood[current_label][j].keys():
					self.likelihood[current_label][j][val] = 1
				else:
					self.likelihood[current_label][j][val] += 1

		# Likelihood Laplace smoothing
		for i in self.classes:
			for j in range(X.shape[1]):
				total = self.prior[i] + len(self.likelihood[i][j]) * self.eps
				#print('total:',i,',',j,',',self.prior[i],',',len(self.likelihood[i][j]))
				for val in self.likelihood[i][j].keys():
					self.likelihood[i][j][val] = (self.likelihood[i][j][val] + self.eps)/total
				self.unknown[i][j] = self.eps/total
		
		for i in self.classes:
			self.prior[i] /= X.shape[0]
	
	def predict(self, X):
		pred = []
		for i in range(X.shape[0]):
			score = dict.fromkeys(self.classes)
			for c in self.classes:
				score[c] = math.log(self.prior[c])
				for j in range(X.shape[1]):
					VAL = X.iloc[i,j]
					if VAL in self.likelihood[c][j].keys():
						score[c] += math.log(self.likelihood[c][j][VAL])
					else:
						score[c] += math.log(self.unknown[c][j])
			result = max(score.items(), key=operator.itemgetter(1))[0]
			pred.append(result)
		return pred

	def accuracy_score(self, X, Y):
		num = len(X)
		acc = 0
		for i in range(num):
			if X[i] == Y[i]:
				acc += 1
		return acc/num