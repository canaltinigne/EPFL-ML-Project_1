import numpy as np

def compute_gradient(y, tx, w):
    e = y - np.dot(tx, w)
    return (-1/y.shape[0])*np.dot(tx.T, e)

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def compute_logistic_gradient(y, tx, w, lambda_=0):
    diff = sigmoid(np.dot(tx, w)) - y
    return np.dot(tx.T, diff) + 2*lambda_ *w