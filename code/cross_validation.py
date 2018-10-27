import numpy as np
from implementations import *
from proj1_helpers import predict_labels
from errors import compute_loss

def build_k_indices(y, k_fold, seed=23):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)

    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]

    return np.array(k_indices)

def split_cross_validation(y, x, k_indices, k):
    """return the loss of ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    X_test, y_test = x[k_indices[k],:], y[k_indices[k]]
    X_train = x[k_indices[[x for x in range(len(k_indices)) if x != k]]].reshape(-1,x.shape[1])
    y_train = y[k_indices[[x for x in range(len(k_indices)) if x != k]]].reshape(-1,1)

    return X_train, y_train, X_test, y_test

def accuracy(pred, y):
    counter = 0
    for i in range(len(pred)):
        if pred[i] == y[i]:
            counter+=1
            
    return counter/len(pred)

def cross_validation(y, X, fold, h_pars={}, model='ridge', seed=23):

    k_indices = build_k_indices(y, fold, seed)
    # define lists to store the loss of training data and test data
    accuracy_tr = []
    accuracy_te = []

    if model == 'ridge':
        for thr in h_pars['threshold']:
            for lambda_ in h_pars['lambda']:
                
                train_err = []
                test_err = []
                
                for k in range(fold):
                    X_train, y_train, X_valid, y_valid = split_cross_validation(y, X, k_indices, k)
                    w, _ = ridge_regression(y_train, X_train, lambda_)
        
                    pred_tr_y = predict_labels(w, X_train, thr)
                    pred_te_y = predict_labels(w, X_valid, thr)

                    train_err.append(accuracy(pred_tr_y, y_train))
                    test_err.append(accuracy(pred_te_y, y_valid))
               
                accuracy_tr.append(np.array([thr, lambda_, np.mean(train_err)]))
                accuracy_te.append(np.array([thr, lambda_, np.mean(test_err)]))

    elif model == 'least':
        
        train_err = []
        test_err = []
        
        for k in range(fold):
            X_train, y_train, X_valid, y_valid = split_cross_validation(y, X, k_indices, k)
            w, _ = least_squares(y_train, X_train)

            pred_tr_y = predict_labels(w, X_train, thr)
            pred_te_y = predict_labels(w, X_valid, thr)

            train_err.append(accuracy(pred_tr_y, y_train))
            test_err.append(accuracy(pred_te_y, y_valid))
        
        accuracy_tr.append(np.array([thr, np.mean(train_err)]))
        accuracy_te.append(np.array([thr, np.mean(test_err)]))

    elif model == 'log':
        initial_w = np.random.normal(0,1,X.shape[1])

        for itr in h_pars['max_iter']:
            for gamma in h_pars['gamma']:
            
                train_err = []
                test_err = []
                
                for k in range(fold):
                    X_train, y_train, X_valid, y_valid = split_cross_validation(real_y, X, k_indices, k)
                    w, loss = logistic_regression(y_train, X_train, initial_w, itr, gamma)

                    train_err.append(loss)
                    test_err.append(compute_loss(y_valid, X_valid, w, t='log'))
                
                accuracy_tr.append(np.array([itr, gamma, np.mean(train_err)]))
                accuracy_te.append(np.array([itr, gamma, np.mean(test_err)]))
             
    return np.array(accuracy_tr), np.array(accuracy_te)