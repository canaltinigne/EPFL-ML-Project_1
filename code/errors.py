import numpy as np
from gradient import sigmoid

def calculate_mse(e):
    return 0.5*np.mean(e**2)

def calculate_mae(e):
    return np.mean(np.abs(e))

def log_loss(y, pred, w, lambda_):
    return -1*np.mean(y*np.log(pred) + (1-y)*np.log(1-pred)) # + lambda_*np.dot(w.T, w) Regularized

def compute_loss(y, tx, w, t='mse', lambda_=0):
    e = y - np.dot(tx, w)

    if t == 'mae':
        return calculate_mae(e)
    elif t == 'log':
        pred = sigmoid(np.dot(tx, w))
        return log_loss(y, pred, w, lambda_)
    else:
        return calculate_mse(e)