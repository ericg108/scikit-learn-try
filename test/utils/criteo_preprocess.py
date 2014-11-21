__author__ = 'nenggong'

import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing

try:
    input_file = sys.argv[1]
except IndexError:
    input_file = '/Users/nenggong/Code/Games/criteo/sample.csv'
try:
    output_file = sys.argv[2]
except IndexError:
    output_file = '/Users/nenggong/Code/Games/criteo/out_sample.csv'

numeric_cols = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13']
category_cols = ['C1', 'C2', 'C5', 'C6', 'C8', 'C9', 'C11', 'C13', 'C14', 'C17', 'C18', 'C19', 'C20',
                 'C22', 'C23', 'C25']
data = pd.read_csv(input_file)
numeric_data = data[numeric_cols]
numeric_data = numeric_data.fillna(numeric_data.mean())
numeric_data_ndarray = numeric_data.as_matrix()
numeric_ndarray_scaled = preprocessing.scale(numeric_data_ndarray)
# print numeric_ndarray_scaled

category_data = data[category_cols]
category_data = category_data.fillna('na')
category_ndarray = category_data.as_matrix()
# print category_ndarray

data_ndarray = np.hstack((numeric_ndarray_scaled, category_ndarray))
# print type(data_ndarray)
# print data_ndarray

np.savetxt(output_file, data_ndarray, delimiter=',', fmt='%s')
# data_ndarray.tofile(output_file,sep=',')