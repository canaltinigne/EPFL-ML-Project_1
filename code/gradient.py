import numpy as np

# Compute gradient for Gradient-based models
def compute_gradient(y, tx, w):
    e = y - np.dot(tx, w)
    return (-1/y.shape[0])*np.dot(tx.T, e)

# Sigmoid function for Logistic Regression
def sigmoid(z):
    return 1. / (1. + np.exp(-z))

# Compute gradient for Logistic Regression models
def compute_logistic_gradient(y, tx, w, lambda_=0):
    diff = sigmoid(np.dot(tx, w)) - y
    return np.dot(tx.T, diff) + 2*lambda_ *w                # Also works for regularized logistic regression when lambda_ > 0