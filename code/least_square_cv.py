from cross_validation import cross_validation
import numpy as np

train_X = np.load("train_X.npy")
train_y = np.load("train_y.npy")

print(train_X.shape)
print(train_y.shape)

np.random.seed(23)
h_pars = {}
h_pars['epoch'] = np.array([1000])
h_pars['l_rate'] = [10**x for x in np.linspace(-3,0,4)]
h_pars['initial_w'] = np.random.normal(0,1,train_X.shape[1])

acc_tr, acc_te = cross_validation(train_y, train_X, 5, h_pars, 'least_GD')

np.save("acc_tr", acc_tr)
np.save("acc_te", acc_te)