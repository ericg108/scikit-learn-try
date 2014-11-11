__author__ = 'nenggong'

import graphlab as gl

''' example from graphlab
url = 'http://s3.amazonaws.com/GraphLab-Datasets/movie_ratings/training_data.csv'
data = gl.SFrame.read_csv(url, column_type_hints={"rating":int})
data.show()
model = gl.recommender.create(data, user_id="user", item_id="movie", target="rating")
results = model.recommend(users=None, k=5)
'''
training_column_types = [str,int,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str]

data_sframe = gl.SFrame.read_csv('/Users/nenggong/Code/Games/avazu/141001.csv', column_type_hints=training_column_types)
features = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id',
            'app_domain', 'app_category', 'device_id', 'device_ip', 'device_os', 'device_make', 'device_model',
            'device_type', 'device_conn_type', 'device_geo_country', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23',
            'C24']

from datetime import datetime

date_format_str = '%y%m%d%H'


def get_hour_from_time(date_str):
    d = datetime.strptime(date_str, date_format_str)
    return {'day': d.day, 'hour': d.hour}


def process_data_frame(data_frame):
    """Split the 'hour' column of a give frame"""
    parsed_date = data_frame['hour'].apply(get_hour_from_time).unpack(column_name_prefix='')
    for col in ['day', 'hour']:
        data_frame[col] = parsed_date[col]


process_data_frame(data_sframe)
env = gl.deploy.environment.Local('hyperparam_search')
training = data_sframe[data_sframe['hour'] <= 20]
validation = data_sframe[data_sframe['hour'] > 20]
training.save('/tmp/training')
validation.save('/tmp/validation')

n_trees = 500
search_space = {
    'max_depth': [5, 10, 15, 20],
    'min_child_weight': [5, 10, 20],
    'step_size': [0.05, 0.1],
    'max_iterations': n_trees
}
# gl.boosted_trees_regression.create()


def parameter_search(training_url, validation_url, default_params):
    job = gl.toolkits.model_parameter_search(env, gl.boosted_trees_regression.create, train_set_path=training_url,
                                             save_path='/tmp/job_output', standard_model_params=default_params,
                                             hyper_params=search_space, test_set_path=validation_url)
    result = gl.SFrame('/tmp/job_output').sort('rmse', ascending=True)
    optimal_params = result[['max_depth', 'min_child_weight']][0]
    optimal_rmse = result['rmse'][0]
    print 'Optimal parameters: %s' % str(optimal_params)
    print 'RMSE: %s' % str(optimal_rmse)
    return optimal_params


new_features = features.remove('hour')
fix_params = {
    'features': new_features,
    'verbose': False
}
fix_params['target'] = 'click'
best_params_click = parameter_search('/tmp/training', '/tmp/validation', fix_params)
best_est = gl.boosted_trees_regression.create(training, features=new_features, target='click', max_iterations=n_trees,
                                   params=best_params_click, verbose=False)

test_column_types = [str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str]
test_data = gl.SFrame.read_csv('/Users/nenggong/Code/Games/avazu/test_rev2', column_type_hints=test_column_types)

pred = best_est.predict(test_data)
def make_submission(prediction, filename='submission.txt'):
    with open(filename, 'w') as f:
        f.write('id,click\n')
        str = test_data['id'] + ',' + pred.astype(str)
        for row in str:
            f.write(row + '\n')

make_submission(pred, filename='submission1.txt')
