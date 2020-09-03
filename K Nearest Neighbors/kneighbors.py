'''
Created by: Nikhil Kohli

This class is the implementation of K-Nearest Neighbors algorithm for Classification from scratch by using only built in python functions. 
I have also included some of the hyperparameters like K and distance metric p.
p = 2 by default which is Euclidean Distance, p=1 is for Manhattan Distance
K is the number of nearest neighbors one wants to consider for prediction

'''

import numpy as np
from collections import Counter
from ml_utils import minkowski


class KNearestNeighbors:
	
	def __init__(self, k=3, p_metric=2):
		self.k = k
		self.p_metric = p_metric


	def fit(self, X_train, Y_train):
		self.X_train = X_train
		self.Y_train = Y_train


	def predict(self, X_Pred):
		Y_Pred = [self.classify(x) for x in X_Pred]
		return np.array(Y_Pred)


	def classify(self, x):
		# calculate distances between data point to all the points in training set
		distances = [minkowski(x, x_train, self.p_metric) for x_train in self.X_train]
		idx_k = np.argsort(distances)[:self.k]	
		k_labels = [self.Y_train[i] for i in idx_k]
		predicted_class = Counter(k_labels).most_common(1)[0][0]

		return predicted_class


