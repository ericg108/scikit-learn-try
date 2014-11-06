__author__ = 'nenggong'

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction import DictVectorizer as dv
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

data_dir = '/Users/nenggong/Code/Games/CTR/'

train_file = data_dir + 'train_sample.tsv'
# test_file = data_dir + 'train_sample.tsv'
test_file = data_dir + 'a.csv'
train = pd.read_csv(train_file)
y = train["label"].as_matrix()

test = pd.read_csv(test_file)
ID = test["Id"]
#print train
# for x in train.iterrows():
#     print x
# for col in train:
#     print col, train[col]
category_cols = ['C1', 'C2', 'C5', 'C6', 'C8', 'C9', 'C11', 'C13', 'C14', 'C17', 'C18', 'C19', 'C20',
                 'C22', 'C23', 'C25']
train_drop_cols = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'label']
test_drop_cols = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'Id']
numeric_cols = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13']


def encoding(data, num_cols, cat_drop_cols):

    num_data = data[num_cols]
    # print num_data
    num_data_fill = num_data.fillna(num_data.mean())
    # print num_data_fill
    num_matrix = num_data_fill.as_matrix()
    # print num_matrix

    x_num_data_array = np.array(num_matrix)
    x_num_scaled_matrix = preprocessing.scale(x_num_data_array)
    # print x_num_scaled_matrix
    # imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    # imp.fit(x_num_data_scaled)
    # print imp
    # x_num_matrix = imp.transform(x_num_data_scaled)
    # print x_num_matrix
    # print cat_matrix

    cat_data = data.drop(cat_drop_cols, axis=1)
    # cat_matrix = data[category_cols].as_matrix()
    cat_data.fillna('NA', inplace=True)
    # cat_matrix = cat_data.as_matrix()
    # print cat_matrix1
    # print type(x_num_scaled_matrix), type(cat_data)
    # print x_num_scaled_matrix
    # print cat_data

    return x_num_scaled_matrix, cat_data

num_train_matrix, cat_train_matrix = encoding(train, numeric_cols, train_drop_cols)

x_cat_train_data = cat_train_matrix.T.to_dict().values()

# num_matrix1 = data.drop(category_cols, axis=1)
# x_num_data = num_matrix1.T.to_dict().values()
# print x_cat_data
# print num_matrix

vectorized = dv(sparse = False)
##NOTE: directly call transform function on training data will cause error since features are not loaded yet
# we should call fit_transform on training data and then transform on test data
# to make sure the test data's features coincide to training data's
vec_x_cat_train = vectorized.fit_transform(x_cat_train_data)
# print vec_x_cat_train, vec_x_cat_train.shape
x_train = np.hstack((num_train_matrix, vec_x_cat_train))


# print x_train, x_train.shape
# print x_test, x_test.shape
# print vectorized

sgd = SGDRegressor()
sgd.fit(x_train, y)
# print sgd.coef_

sgd = LogisticRegression()
sgd.fit(x_train, y)
# print sgd.coef_
x_train = None

num_test_matrix, cat_test_matrix = encoding(test, numeric_cols, numeric_cols)
x_cat_test_data = cat_test_matrix.T.to_dict().values()
vec_x_cat_test = vectorized.transform(x_cat_test_data)
# print vec_x_cat_test, vec_x_cat_test.shape
x_test = np.hstack((num_test_matrix, vec_x_cat_test))

prob =  sgd.predict_proba(x_test)
# print prob
# print type(prob[:,1]), prob[:,1].shape

rows, = prob[:,1].shape
for i in range(rows):
    str = "%s,%s" %(ID[i], prob[i,1])
    print str
# print sgd.predict(x_test)

# can only process categorical col whose value is integer
# enc = preprocessing.OneHotEncoder()
#print enc.fit(cat_matrix)
# cat_matrix.fillna('NA', inplace=True)

