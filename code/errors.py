import numpy as np
from gradient import sigmoid

def calculate_mse(e):
    return 0.5*np.mean(e**2)

def calculate_mae(e):
    return np.mean(np.abs(e))

def calculate_logistic_loss(y, tx, w, lambda_):
    pred = sigmoid(np.dot(tx, w))
    return -np.mean(y*np.log(pred) + (1-y)*np.log(1-pred)) + lambda_*np.dot(w.T, w)

def compute_loss(y, tx, w, t='mse', lambda_=0):
    e = y - np.dot(tx, w)

    if t == 'mae':
        return calculate_mae(e)
    elif t == 'logistic':
        return calculate_logistic_loss(y, tx, w, lambda_)
    else:
        return calculate_mse(e)