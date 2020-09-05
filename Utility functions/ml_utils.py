"""
Created by: Nikhil Kohli

These functions serves as the utility functions used by various Machine Learning Algorithms.
Instead of creating each functions along with the ML algorithm class, I am maintaining a separate code file for all helper functions which I might use ahead. 

"""

import numpy as np

# Instead of using sklearn function, lets create our own train test split function. Super easy to do! 
def train_test_split(X, Y, test_size=0.3, random_seed=1234):
	#setting the seed for reproducibility	
	np.random.seed(random_seed)

	train_size = int((1-test_size) * (X.shape[0])) 
	indices = list(range(X.shape[0]))
	#shuffle indices to create a random split
	np.random.shuffle(indices)
	#split indices into train and test
	train_indices = indices[:train_size]
	test_indices = indices[train_size:]

	#split the actual data using the train test indices
	X_train, X_test = X[train_indices], X[test_indices]
	Y_train, Y_test = Y[train_indices], Y[test_indices]
	
	return X_train, X_test, Y_train, Y_test



# Distance metric for K-Nearest Neighbors
def minkowski(Q, R, p):
	'''
	This function calculates the distance between 2 vectors
	p=1 will give Manhattan distance
	p=2 will give Euclidean distance
	'''
	distance = (np.sum(np.abs(Q-R)**p)**(1/p))
	return distance


def accuracy(Y_true, Y_pred):
	acc = np.sum(Y_true == Y_pred) / len(Y_true)
	return acc 

def mse(Y_true, Y_pred):
	mse = np.mean(np.square(Y_true - Y_pred))
	return mse


