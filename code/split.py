import numpy as np 

def split_data(x, y, ratio, seed=1):
    
    np.random.seed(seed)
    element_num = int(np.floor(ratio * len(y)))
    random_ind = np.random.permutation(len(y))
    
    ind_train = random_ind[:element_num]
    ind_test = random_ind[element_num:]

    x_tr = x[ind_train]
    x_te = x[ind_test]
    y_tr = y[ind_train]
    y_te = y[ind_test]
    
    return x_tr, y_tr, x_te, y_te