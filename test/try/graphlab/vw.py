__author__ = 'nenggong'

import graphlab as gl

dataBase = '/Users/nenggong/Code/Games/'
training_column_type = [str, int, int, int, int, int, int, int, int, int, int, int, int, int, int, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str]
target_name = 'Label'
training_data_frame = gl.SFrame.read_csv(dataBase + 'training_data', column_type_hints=training_column_type)

def transform_target(data_frame, target):
    data_frame[target] = data_frame[target].apply(lambda x: 1 if x == 1 else -1)

transform_target(training_data_frame, target_name)

raw_lr = gl.vowpal_wabbit.create(training_data_frame, target_name, loss_function='logistic', max_iterations=100, step_size=0.1)

validation_data_frame = gl.SFrame.read_csv(dataBase + 'validation_data', column_type_hints=training_column_type)
transform_target(validation_data_frame, target_name)

raw_lr_evaluation = raw_lr.evaluate(validation_data_frame)
print raw_lr_evaluation['confusion_matrix']
print raw_lr_evaluation['accuracy']

l1_lr = gl.vowpal_wabbit.create(training_data_frame, target_name, loss_function='logistic', max_iterations=100, step_size=0.1, l1_penalty=0.1)
l1_lr_evaluation = raw_lr.evaluate(validation_data_frame)
print l1_lr_evaluation['confusion_matrix']
print l1_lr_evaluation['accuracy']
