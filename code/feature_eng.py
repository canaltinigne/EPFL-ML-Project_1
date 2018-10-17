import numpy as np

def pca(X,deg):
    X = X - np.mean(X,axis=0)
    cov = np.cov(X.T)
    eig_value, eig_vec = np.linalg.eig(cov)
    sort_idx = np.argsort(-eig_value)
    r,c = eig_vec.shape
    sorted_vecs = np.ones((r, deg))

    for i in range(deg):
        sorted_vecs[:,i]=eig_vec[:,sort_idx[i]]

    reduced = np.dot(X,sorted_vecs)

    return reduced