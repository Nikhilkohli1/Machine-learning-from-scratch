'''
Created by: Nikhil Kohli

This class is the implementation of K-Nearest Neighbors algorithm for Classification from scratch by using only built in python functions. 
I have also included some of the hyperparameters like K and distance metric p.
p = 2 by default which is Euclidean Distance, p=1 is for Manhattan Distance
K is the number of nearest neighbors one wants to consider for prediction

'''

class LinearRegression:

	def __init__(self, alpha=0.0001, num_iters=2000):
		self.alpha = alpha
		self.num_iters = num_iters
		self.weights = None
		self.bias = None

	def fit(self, X, Y):
		n_samples, n_features = X.shape

		self.weights = np.zeros(n_features)
		self.bias = 0 

		#implement gradient descent for updating the weights and bias
		for i in num_iters:
			Y_pred = np.dot(X, self.weights) + self.bias

			##compute the gradients 
			dw = (1/n_samples) * np.dot(X.T, (Y_pred - Y))
			db = (1/n_samples) * np.sum(Y_pred - Y)

			self.weights -= self.alpha * dw
			self.bias -+ self.alpha * db





	def predict(self, X):
		Y_pred = np.dot(X, self.weights) + self.bias

