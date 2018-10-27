# Regression and Classification Models

import numpy as np
from gradient import compute_logistic_gradient
from gradient import compute_gradient
from errors import compute_loss

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    
    for n_iter in range(max_iters):
        dw = compute_gradient(y, tx, w)
        w = w - gamma*dw
        loss = compute_loss(y, tx, w)

    return (w, loss)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w

    for n_iter in range(max_iters):
        rnd_i = np.random.randint(0, len(y))
        dw = compute_gradient(y[rnd_i], tx[rnd_i], w)
        w = w - gamma*dw
        loss = compute_loss(y, tx, w)

    return (w, loss)

def least_squares(y, tx):
    w = np.dot(np.dot(np.linalg.inv(np.dot(tx.T, tx)), tx.T), y)
    loss = compute_loss(y, tx, w)

    return (w, loss)

def ridge_regression(y, tx, lambda_):
    w = np.dot(np.dot(np.linalg.inv(np.dot(tx.T, tx) + lambda_*2*len(y)*np.identity(tx.shape[1])), tx.T), y)
    loss = compute_loss(y, tx, w)
    
    return (w, loss)

def logistic_regression(y, tx, initial_w, max_iters, gamma):    # SGD
 
    w = initial_w
    losses = []
    
    for n_iter in range(max_iters):
        dw = compute_logistic_gradient(y, tx, w)
        w = w - gamma*dw
        loss = compute_loss(y, tx, w, t='log')
        losses.append(loss)

        if n_iter%100 == 0:
            print("100 iter completed")
        #print("iter: {} - loss: {}".format(n_iter, loss))

    return (w, losses)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    
    for n_iter in range(max_iters):
        dw = compute_logistic_gradient(y, tx, w, lambda_)
        w = w - gamma*dw
        loss = compute_loss(y, tx, w, 'logistic', lambda_)

    return (w, loss)