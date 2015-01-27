__author__ = 'nenggong'

from math import sqrt

trainFileName = '../data/hw2_adaboost_train.dat'

def read_training_data(trainFileName):
    train_data = []
    with open(trainFileName, 'r') as trainFile:
        for line in trainFile:
            # print line.strip()
            raw_data = line.strip().split(' ')
            data_arr = []
            for i in range(len(raw_data) - 1):
                data_arr.append(float(raw_data[i]))
            data_arr.append(int(raw_data[-1]))
            train_data.append(data_arr)
    return train_data

def get_best_stump(train_data, ith_feature, u):
    sorted_train_data_by_col = []
    sorted_row_indices = []
    y_col = [row[-1] for row in train_data]
    for i in range(len(train_data[0]) - 1):
        sorted_train_data_col = []
        sorted_row_index = []
        for index, num in sorted(enumerate([row[i] for row in train_data]), key=lambda x: x[1]):
            sorted_row_index.append(index)
            sorted_train_data_col.append(num)
        sorted_train_data_by_col.append(sorted_train_data_col)
        sorted_row_indices.append(sorted_row_index)
    # print y_col
    # print sorted_row_indices
    # print sorted_train_data_by_col
    threshold_num = len(y_col)
    min_err = threshold_num
    err = 0
    theta = 0
    s_min = 1
    for s in (-1, 1):
        for i in range(threshold_num):
            err += u[i] * s * ((1 if sorted_train_data_by_col[ith_feature][i] >= sorted_train_data_by_col[ith_feature][0] else -1) == y_col[i])
        if err < min_err:
            min_err = err
            theta = sorted_train_data_by_col[ith_feature][0] -1
            s_min = s
        for i in range(threshold_num - 1):
            ith_theta = (sorted_train_data_by_col[ith_feature][i] + sorted_train_data_by_col[ith_feature][i + 1]) << 1
            y_pre = 1 if sorted_train_data_by_col[ith_feature][i] > ith_theta else -1
            if y_pre == y_col[i]:
                err -= u[i]
                if err < min_err:
                    min_err = err
                    theta = ith_theta
                    s_min = s
            else:
                err += u[i]
    #update u
    eps = sqrt((1-min_err)/min_err)
    for i in range(threshold_num):
        y_pre = 1 if sorted_train_data_by_col[ith_feature][i] > ith_theta else -1
        if y_pre == y_col[i]:
            u[i] /= eps
        else:
            u[i] *= eps
    return s_min, theta, u





def adaboost_stump(train_data, stump_number):
    stumps = []
    features = len(train_data[0]) - 1
    samples = len(train_data)
    u = [1/samples] * samples
    for i in range(stump_number):
        ith_feature = i % features
        s, theta, uu= get_best_stump(train_data, ith_feature, u)
        u = uu
        stumps.append((s, i, theta))
    return stumps



train_data = read_training_data(trainFileName)
get_best_stump(train_data, 1, 1)
# stumps = adaboost_stump(train_data, stump_number=300)

