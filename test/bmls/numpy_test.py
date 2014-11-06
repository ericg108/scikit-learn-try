__author__ = 'nenggong'

from math import pi
import numpy as np

a = np.arange(15).reshape(3,5)
print a

aa = np.arange(0, 2, 0.3, dtype=np.float64)
print aa

b = np.array([1, 3, 5])
print b.dtype, b.size, b.ndim, b

zero = np.zeros((2,3))
print zero

empty = np.empty((2,3))
print empty

x = np.linspace( 0, 2*pi, 100 )
#print x
fx = np.sin(x)
#print fx

c = np.random.random((2,3))
print c, c.sum(), c.min(), c.max()
print c.sum(axis=0), c.min(axis=1)

def f(x,y):
    return 10*x +y
ff = np.fromfunction(f,(5,4),dtype=int)
#print ff
# for row in ff:
#     print row
# for ele in ff.flat:
#     print ele
print ff.transpose()