# Simply the copy of Test Set Prediction Notebook

# It takes nearly 20 seconds be patient

import numpy as np
from cross_validation import cross_validation
from polynomial import build_poly
from implementations import ridge_regression
from proj1_helpers import load_csv_data
from proj1_helpers import predict_labels
from proj1_helpers import create_csv_submission

USE_PRETRAINED_WEIGHTS = False

# Read the test set
test_set = load_csv_data('../data/test.csv')
y_test, X_test, ids, columns = test_set

# Selected columns
selected_features = np.array([1,3,9,10,11,13,21,22,23])
selected_features = np.sort(np.append(selected_features, [0,4,5,6,12]))

# Log transformed columns
log_transformed_columns = [3,9,10,13,21]

for i in log_transformed_columns:
    X_test[np.where(X_test[:,i] != -999), i] = np.log(X_test[np.where(X_test[:,i] != -999),i] + 1)
    X_test[np.where(X_test[:,i] == -999)] = -999

# Select the correct features
X_test = X_test[:, selected_features]

# Standartize the data
X_train = np.load('X_train_not_normalized.npy')

for i in [x for x in range(X_train.shape[1]) if x != 12]:
    col_val = X_train[np.where(X_train[:,i] != -999), i]
    X_test[np.where(X_test[:,i] != -999), i] = (X_test[np.where(X_test[:,i] != -999), i] - np.mean(col_val)) / (np.std(col_val)) 

# Data Imputation w/ median for DER_mass_MMC feature
X_test[np.where(X_test[:,0] == -999), 0] = np.median(X_test[np.where(X_test[:,0] != -999), 0])

# Split 3 different subsets
ids_pri_0 = np.where(X_test[:, -2] == 0)[0]
ids_pri_1 = np.where(X_test[:, -2] == 1)[0]
ids_pri_23 = np.where((X_test[:, -2] == 2) | (X_test[:, -2] == 3))[0]

X_pri_0 = X_test[(X_test[:, -2] == 0),:]
X_pri_1 = X_test[(X_test[:, -2] == 1),:]
X_pri_23 = X_test[(X_test[:, -2] == 2) | (X_test[:, -2] == 3),:]

# Removing PRI_jet_num feature
X_pri_0 = np.delete(X_pri_0, np.s_[12], axis=1)
X_pri_1 = np.delete(X_pri_1, np.s_[12], axis=1)
X_pri_23 = np.delete(X_pri_23, np.s_[12], axis=1)

# Remove -999 columns for first two subsets
selected_features = np.array([1,3,9,10,11,13,21,23])
selected_features = np.sort(np.append(selected_features, [0,4,5,6,12]))

delete_columns_0 = []

for i in range(X_pri_0.shape[1]):
    if np.isin(True, (X_pri_0[:,i] == -999)):
        delete_columns_0.append(i)
        
delete_columns_1 = []

for i in range(X_pri_1.shape[1]):
    if np.isin(True, (X_pri_1[:,i] == -999)):
        delete_columns_1.append(i)
        
X_pri_0 = np.delete(X_pri_0, np.s_[delete_columns_0], axis=1)  
X_pri_1 = np.delete(X_pri_1, np.s_[delete_columns_1], axis=1)  

# Predict the test set
predictions = np.zeros(len(y_test))

w0 = 0
w1 = 0
w23 = 0

X_train_pri_0 = np.load('X_pri_0.npy')
X_train_pri_1 = np.load('X_pri_1.npy')
X_train_pri_23 = np.load('X_pri_23.npy')

y_train_pri_0 = np.load('y_pri_0.npy')
y_train_pri_1 = np.load('y_pri_1.npy')
y_train_pri_23 = np.load('y_pri_23.npy')

if USE_PRETRAINED_WEIGHTS == True:
    w0 = np.load('w0.npy')
    w1 = np.load('w1.npy')
    w23 = np.load('w23.npy')
else:                                                                                           # Model trained here
    w0, loss0 = ridge_regression(y_train_pri_0, build_poly(X_train_pri_0, 12), 1e-14)
    w1, loss1 = ridge_regression(y_train_pri_1, build_poly(X_train_pri_1, 12), 1e-3)
    w23, loss23 = ridge_regression(y_train_pri_23, build_poly(X_train_pri_23, 11), 1e-5)

pri_0_y = predict_labels(w0, build_poly(X_pri_0, 12))
pri_1_y = predict_labels(w1, build_poly(X_pri_1, 12))
pri_23_y = predict_labels(w23, build_poly(X_pri_23, 11))

predictions[ids_pri_0] = pri_0_y
predictions[ids_pri_1] = pri_1_y
predictions[ids_pri_23] = pri_23_y

create_csv_submission(ids, predictions, 'output.csv')

