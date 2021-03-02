import numpy as np
from math import exp
from math import log

#sigmoid function
beta=np.random.randn(5)
def sigmoid(u):
	expu=np.exp(u)
	return expu/(1+expu)
print("sigmoid function: ", sigmoid)


#softmax function
p1 = exp(1) / (exp(1) + exp(3) + exp(2))
p2 = exp(3) / (exp(1) + exp(3) + exp(2))
p3 = exp(2) / (exp(1) + exp(3) + exp(2))
print("softmax function: ")
print(p1, p2, p3)
print(p1 + p2 + p3)

# Cross entropy function
def CrossEntropy(yHat, y):
    if y == 1:
    	return -log(yHat)
    else:
    	return -log(1 - yHat)

#Binary cross entropy
def binary_cross_entropy(actual, predicted):
	sum_score = 0.0
	for i in range(len(actual)):
		sum_score += actual[i] * log(1e-15 + predicted[i])
		mean_sum_score = 1.0 / len(actual) * sum_score
		return -mean_sum_score
print(binary_cross_entropy([1, 0, 1, 0], [1, 1, 1, 0]))


#Multi-classification
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

X, y1 = make_classification(n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1)
y2 = shuffle(y1, random_state=1)
y3 = shuffle(y1, random_state=2)
Y = np.vstack((y1, y2, y3)).T
n_samples, n_features = X.shape 
n_outputs = Y.shape[1] 
n_classes = 3
forest = RandomForestClassifier(random_state=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
print(multi_target_forest.fit(X, Y).predict(X))