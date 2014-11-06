__author__ = 'nenggong'

import numpy as np
from math import sqrt


A = [[0, 1, 1, 0, 0, 0],
     [1, 0, 0, 1, 0, 1],
     [1, 0, 0, 1, 0, 0],
     [0, 1, 1, 0, 1, 0],
     [0, 0, 0, 1, 0, 1],
     [0, 1, 0, 0, 1, 0]
     ]

lamda, v = np.linalg.eig(np.array(A))

print lamda
print v[:,1]
print v[:, 1]/sqrt(sum(v[:, 1] ** 2))
print sorted(v[:, 1])
print sum(v[:, 1])