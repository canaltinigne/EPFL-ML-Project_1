import numpy as np
from gradient import sigmoid

def calculate_mse(e):
    return 0.5*np.mean(e**2)

def calculate_mae(e):
    return np.mean(np.abs(e))

def log_loss(y, tx, w, lambda_):
    pred = sigmoid(np.dot(tx, w))
    return -1*np.mean(np.add(np.multiply(y,np.log(pred)), np.multiply((1-y),np.log(1-pred)))) + lambda_*np.dot(w.T, w)

def compute_loss(y, tx, w, t='mse', lambda_=0):
    if t == 'mae':
        return calculate_mae(y - np.dot(tx, w))
    elif t == 'log':
        return log_loss(y, tx, w, lambda_)
    else:
        return calculate_mse(y - np.dot(tx, w))