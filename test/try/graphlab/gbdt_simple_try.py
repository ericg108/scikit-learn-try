__author__ = 'nenggong'

import graphlab as gl
from math import log

''' example from graphlab
url = 'http://s3.amazonaws.com/GraphLab-Datasets/movie_ratings/training_data.csv'
data = gl.SFrame.read_csv(url, column_type_hints={"rating":int})
data.show()
model = gl.recommender.create(data, user_id="user", item_id="movie", target="rating")
results = model.recommend(users=None, k=5)
'''


def log_loss(y_pred, y_label):
    epsilon = 1e-15
    p = min(max(y_pred, epsilon), 1. - epsilon)
    return -log(p) if y_label == 1. else -log(1. - p)


dataBasePath = '/Users/nenggong/Code/Games/avazu/'
training_column_types = [str, int, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str,
                         str, str, str, str, str]

data_sframe = gl.SFrame.read_csv(dataBasePath + 'train', column_type_hints=training_column_types)

features = ['id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id',
            'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type',
            'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
new_features = features.remove('hour')
n_trees = 500
best_est = gl.boosted_trees_regression.create(data_sframe, features=new_features, target='click',
                                              max_iterations=n_trees, max_depth=5, step_size=0.1, min_child_weight=10, )
# best_est.save('/tmp/best_estimator')
# best_est_loaded = gl.load_model('/tmp/best_estimator')

def validate():
    global validation_data, validation_pred
    validation_data = gl.SFrame.read_csv(dataBasePath + 'validation', column_type_hints=training_column_types)
    validation_pred = best_est.predict(validation_data)
    validation_rmse = best_est.evaluate(validation_data, 'rmse')
    print validation_rmse

# validate()


def compute_total_logloss(prediction):
    total_logloss = 0.0
    for i, prob in enumerate(prediction):
        print i, validation_data['id'][i], prob
        total_logloss += log_loss(validation_data['id'][i], prob)
    return total_logloss


# validation_log_loss = compute_total_logloss(validation_pred)
# print validation_log_loss


def predict_test_data():
    global test_column_types, test_data, pred
    test_column_types = [str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str,
                         str, str, str, str, str]
    test_data = gl.SFrame.read_csv(dataBasePath + 'test', column_type_hints=test_column_types)
    pred = best_est.predict(test_data)


predict_test_data()

def make_submission(pred, filename='submission.txt'):
    with open(filename, 'w') as f:
        f.write('id,click\n')
        rows = test_data['id'] + ',' + pred.astype(str)
        for row in rows:
            f.write(row + '\n')

make_submission(pred, filename='submission1.txt')
