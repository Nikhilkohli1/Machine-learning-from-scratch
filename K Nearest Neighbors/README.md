# K-Nearest Neighbors - Implementation from Scratch  

K-Nearest Neighbors is a non-probabilistic Supervised learning algorithm i.e. it does not give a probability to be in a class but rather classifies the data on hard assignment. KNN uses distance metrics to find similarities. 

- KNN find the 'k' nearest data points to the query point for which the prediction is to be made, and then do a 'Majority vote' or also refered to as 'Plurality vote' to predict the category. In regression, it takes the average value of the k nearest neighbors. 

- KNN is a type of lazy learning, where the function is only approximated locally and all computation is deferred until function evaluation. Since this algorithm relies on distance for classification, normalizing the training data can improve its accuracy dramatically.

- Distance metric used is 'minkowski' which further drills down to few types of distance functions depending upon the value of the hyperparameter 'p'. 

	p=1 corresponds to Manhattan Distance
	p=2 corresponds to Euclidean Distance

- It also has a parameter weights which by default is 'uniform' but can be set to 'distance'. This assigns weights to the contributions of the neighbors, so that the nearer neighbors contribute more to the average than the more distant ones. For example, a common weighting scheme consists in giving each neighbor a weight of 1/d, where d is the distance to the neighbor.


***I will implement KNN along with these few hyperparameters from scratch to fully understand KNN with its underlining Mathematics and then compare it with sklearn implementation on a real world dataset ***
