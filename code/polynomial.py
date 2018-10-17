import numpy as np

def build_poly(x, degree):
    temp_arr = np.ones(len(x))
    for i in range(1, degree+1):
        arr = np.power(x, i)
        temp_arr = np.column_stack((temp_arr, arr))
        
    return temp_arr