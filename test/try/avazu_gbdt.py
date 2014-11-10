__author__ = 'nenggong'

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

data_file = '/Users/nenggong/Code/Games/avazu/14100100.csv'
data = np.loadtxt(data_file, dtype=np.str, delimiter=',', skiprows=1)
# print data.shape, type(data)

y = data[:, 1]
x_train, x_test, y_train, y_test = train_test_split(data, y)

from sklearn.grid_search import GridSearchCV

param_grid = {
    'learning_rate': [0.1, 0.05, 0.02, 0.01],
    'max_depth': [4, 6, 8],
    'min_samples_leaf': [3, 5, 9, 17],
    'subsample': [0.5, 0.7]
    'max_features': [1.0, 0.5, 0.3, 0.1]
}
est = GradientBoostingRegressor(n_estimators=500, loss='ls')
gs_cv = GridSearchCV(est, param_grid, n_jobs=-1).fit(x_train[:,2:], y_train) #n_jobs denotes running jobs in parallel
print gs_cv.best_params_
print log_loss(y_test, gs_cv.predict_proba(x_test))


