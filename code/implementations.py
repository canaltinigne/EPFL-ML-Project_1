# Regression and Classification Models

import numpy as np
from gradient import compute_logistic_gradient
from gradient import compute_gradient
from errors import compute_loss

# Linear regression with GD
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w.reshape(-1,1)
    y = y.reshape(-1,1)
    
    for n_iter in range(max_iters):
        dw = compute_gradient(y, tx, w)
        w = w - gamma*dw
        loss = compute_loss(y, tx, w)

    return (w, loss)

# Linear regression with SGD
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w.reshape(-1,1)
    y = y.reshape(-1,1)
    np.random.seed(23)

    for n_iter in range(max_iters):
        rnd_i = np.random.randint(0, len(y))                # Choose random element in the dataset
        dw = compute_gradient(y[rnd_i], tx[rnd_i,:].reshape(1,-1), w)
        w = w - gamma*dw
        loss = compute_loss(y, tx, w)

    return (w, loss)

# Least squares 
def least_squares(y, tx):
    w = np.dot(np.dot(np.linalg.inv(np.dot(tx.T, tx)), tx.T), y)
    loss = compute_loss(y, tx, w)

    return (w, loss)

# Ridge regression 
def ridge_regression(y, tx, lambda_):
    w = np.dot(np.dot(np.linalg.inv(np.dot(tx.T, tx) + lambda_*2*len(y)*np.identity(tx.shape[1])), tx.T), y)
    loss = compute_loss(y, tx, w)
    
    return (w, loss)

# Logistic regression
def logistic_regression(y, tx, initial_w, max_iters, gamma):    # GD
    w = initial_w.reshape(-1,1)
    y = y.reshape(-1,1)

    for n_iter in range(max_iters):
        dw = compute_logistic_gradient(y, tx, w)
        w = w - gamma*dw
        loss = compute_loss(y, tx, w, t='log')

    return (w, loss)

# Regularized logistic regression
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w.reshape(-1,1)
    y = y.reshape(-1,1)

    for n_iter in range(max_iters):
        dw = compute_logistic_gradient(y, tx, w, lambda_)
        w = w - gamma*dw
        loss = compute_loss(y, tx, w, 'log', lambda_)

    return (w, loss)