import numpy as np

def compute_gradient(y, tx, w):
    e = y - np.dot(tx, w)
    return (-1/len(y))*np.dot(tx.T, e)

def calculate_mse(e):
    return 0.5*np.mean(e**2)

def calculate_mae(e):
    return np.mean(np.abs(e))

def calculate_logistic_loss(y, tx, w, lambda_):
    pred = sigmoid(np.dot(tx, w))
    return -np.mean(y*np.log(pred) + (1-y)*np.log(1-pred)) + lambda_*np.dot(w.T, w)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y, tx, w, t='mse', lambda_=0):
    e = y - np.dot(tx, w)

    if t == 'mae':
        return calculate_mae(e)
    elif t == 'logistic':
        return calculate_logistic_loss(y, tx, w, lambda_)
    else:
        return calculate_mse(e)

def compute_logistic_gradient(y, tx, w, lambda_=0):
    return np.dot(tx.T, (sigmoid(np.dot(tx, w)-y))) + 2*lambda_ *w

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

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    
    for n_iter in range(max_iters):
        dw = compute_logistic_gradient(y, tx, w)
        w = w - gamma*dw
        loss = compute_loss(y, tx, w, t='logistic')

    return (w, loss)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    
    for n_iter in range(max_iters):
        dw = compute_logistic_gradient(y, tx, w, lambda_)
        w = w - gamma*dw
        loss = compute_loss(y, tx, w, t='logistic', lambda_)

    return (w, loss)