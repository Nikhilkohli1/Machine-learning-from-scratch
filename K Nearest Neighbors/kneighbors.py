'''
This class is the implementation of K Nearest Neighbors algorithm for Classification. 
I will also include some of the hyperparameters like weights and distance metric.
'''

import numpy 


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


