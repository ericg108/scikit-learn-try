__author__ = 'nenggong'


import numpy as np
from sklearn import cross_validation

x = np.random.rand(10,2)
print x
y = np.random.randint(2, size=10)
print y

kf = cross_validation.KFold(10, n_folds=3)
print "KFold: ", len(kf)
for train_idx, test_idx in kf:
    print "TRAIN:", train_idx, "TEST:", test_idx

skf = cross_validation.StratifiedKFold(y, n_folds=3)
print "StratifiedKFold: ", len(skf)
for train_idx, test_idx in skf:
    print "TRAIN:", train_idx, "TEST:", test_idx

ss = cross_validation.ShuffleSplit(10, n_iter=4, test_size=0.3)
print "ShuffleSplit: ", len(ss)
for train_idx, test_idx in ss:
    print "TRAIN:", train_idx, "TEST:", test_idx

# print np.newaxis