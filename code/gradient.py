import numpy as np

def compute_gradient(y, tx, w):
    e = y - np.dot(tx, w)
    return (-1/len(y))*np.dot(tx.T, e)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_logistic_gradient(y, tx, w, lambda_=0):
    return np.dot(tx.T, (sigmoid(np.dot(tx, w)-y))) + 2*lambda_ *w