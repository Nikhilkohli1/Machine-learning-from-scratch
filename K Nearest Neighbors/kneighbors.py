'''
This class is the implementation of K Nearest Neighbors algorithm for Classification. 

I will also include some of the hyperparameters like weights and distance metric p.
p = 2 by default which is Euclidean Distance, p=1 is Manhattan Distance
Weights can values 'uniform' or 'distance' where each neighbor will have an impact on the classification based on how closer they are in terms of the distance metric
K is the number of nearest neighbors one wants to consider 

'''

import numpy 
from collections import Counter


class KNearestNeighbors():
	
	def _init_(self, k=3, weights='uniform', p_metric=2):
		self.k = k
		self.weights = weights
		self.p_metric = p_metric



	def fit(self, X_train, Y_train):
		self.X_train = X_train
		self.Y_train = Y_train


	def predict(self, X_Pred):
		Y_Pred = [classify(x) for x in X_Pred]
		return Y_Pred.toarray()


	def classify():
		# calculate distances from data point to all the points in training set
		distances = [minkowski(X, x_train) for x_train in X_train]

		idx = np.argsort(distances)[:self.k]	

		labels = [X_train[i] for i in idx]

		predicted_class = Counter(labels).most_common(1)[0][0]

		return predicted_class


