import numpy as np

# Build polynomial dataset with the given degree
def build_poly(X, degree):

    temp_arr = np.ones(X.shape[0])

    for i in range(X.shape[1]):
        for j in range(2, degree+1):
            new_col = X[:,i]**j
            temp_arr = np.column_stack((temp_arr, new_col))
      
    return np.column_stack((X, temp_arr))