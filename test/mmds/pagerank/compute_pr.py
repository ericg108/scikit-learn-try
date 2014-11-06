__author__ = 'nenggong'

import numpy as np

m = [[0,0,0], [0.5, 0, 0], [0.5, 1, 1]]
m = np.asmatrix(m,dtype=float)
print m * 0.7 + 0.3*1/3

m = [[0,0,1], [0.5, 0, 0], [0.5, 1, 0]]
m = np.asmatrix(m,dtype=float)
a = m * 0.85 + 0.15*1/3
print a.shape

x = [1, 1, 1]
x = np.asmatrix(x).transpose()
print x.shape
for i in range(0,6):
    print i
    x = np.dot(a, x)
    print x