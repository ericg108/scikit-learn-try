#Refer to http://www.52ml.net/16044.html
#and http://www.slideshare.net/DataRobot/gradient-boosted-regression-trees-in-scikitlearn
__author__ = 'nenggong'

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.cross_validation import train_test_split

X, y = make_hastie_10_2(n_samples=5000)

x_train, x_test, y_train, y_test = train_test_split(X, y)
est = GradientBoostingClassifier(n_estimators=200, max_depth=3)
est.fit(x_train, y_train)

# pred = est.predict(x_test)
# acc = est.score(x_test, y_test)
# print 'ACC: %.4f' %acc



import numpy as np

def ground_truth(x):
    return x * np.sin(x) + np.sin(2 * x)

def gen_data(n_samples=200):
    np.random.seed(13)
    x = np.random.uniform(0, 10, size=n_samples)
    x.sort()
    y = ground_truth(x) + 0.75 * np.random.normal(size=n_samples)
    # print x, type(x)
    # print y, type(y)
    train_mask = np.random.randint(0, 2, size=n_samples).astype(np.bool)
    x_train, y_train= x[train_mask, np.newaxis], y[train_mask]
    x_test, y_test= x[~train_mask, np.newaxis], y[~train_mask]
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = gen_data(n_samples=200)
# print x_train, x_test, y_train, y_test
import matplotlib.pyplot as plt

x_plot = np.linspace(0, 10, num=500)
# print x_plot[:, np.newaxis]
def plot_data(figsize=(8, 5)):
    fig = plt.figure(figsize=figsize)
    gt = plt.plot(x_plot, ground_truth(x_plot), alpha=0.4, label='ground truth')

    #plot training and testing data
    plt.scatter(x_train, y_train, s=10, alpha=0.4)
    plt.scatter(x_test, y_test, s=10, alpha=0.4, color='red')
    plt.xlim((0, 10))
    plt.ylabel('y')
    plt.xlabel('x')
    # plt.show()


def plot_tree_regression():
    # global DecisionTreeRegressor, est
    plot_data(figsize=(8, 5))
    from sklearn.tree import DecisionTreeRegressor

    est = DecisionTreeRegressor(max_depth=1).fit(x_train, y_train)
    plt.plot(x_plot, est.predict(x_plot[:, np.newaxis]), label='RT max_depth=1', color='g', alpha=0.9, linewidth=2)
    est = DecisionTreeRegressor(max_depth=3).fit(x_train, y_train)
    plt.plot(x_plot, est.predict(x_plot[:, np.newaxis]), label='RT max_depth=1', color='g', alpha=0.9, linewidth=1)


# plot_tree_regression()


est = GradientBoostingRegressor(n_estimators=1000, max_depth=1, learning_rate=1.0)
est.fit(x_train, y_train)


def staged_predict_plot():
    # global ax, islice, first, pred
    ax = plt.gca()
    from itertools import islice

    first = True
    for pred in islice(est.staged_predict(x_plot[:, np.newaxis]), 0, 1000, 10):
        # print pred
        plt.plot(x_plot, pred, color='r', alpha=0.2)
        if first:
            ax.annotate('High bias - low variance', xy=(x_plot[x_plot.shape[0] // 2], pred[x_plot.shape[0] // 2]),
                        xycoords='data',
                        xytext=(3, 4), textcoords='data', arrowprops=dict(arrowstyle="-", connectionstyle="arc"))
            first = False
    pred = est.predict(x_plot[:, np.newaxis])
    plt.plot(x_plot, pred, color='r', label='GBRT max_depth=1')
    ax.annotate('Low bias - high variance', xy=(x_plot[x_plot.shape[0] // 2], pred[x_plot.shape[0] // 2]),
                xycoords='data',
                xytext=(6.25, -6), textcoords='data', arrowprops=dict(arrowstyle="-", connectionstyle="arc"))
    plt.legend(loc='upper left')


# staged_predict_plot()

n_estimators = len(est.estimators_)
def deviance_plot(est, x_test, y_test, ax=None, label='', train_color='#2c7bb6', test_color='#d7191c', alpha=1.0):
    test_deviance = np.empty(n_estimators)
    for i, pred in enumerate(est.staged_predict(x_test)):
        test_deviance[i] = est.loss_(y_test, pred)
    if ax is None:
        fig = plt.figure(figsize=(8,5))
        ax = plt.gca()
    ax.plot(np.arange(n_estimators)+1, test_deviance, color=test_color, label='Test %s' % label, linewidth=2, alpha=alpha)
    ax.plot(np.arange(n_estimators)+1, est.train_score_, color=train_color, label='Train %s' % label, linewidth=2, alpha=alpha)
    ax.set_ylabel('Error')
    ax.set_xlabel('n_estimators')
    return ax, test_deviance

ax, test_deviance = deviance_plot(est, x_test, y_test)
ax.annotate('Lowest test error', xy=(test_deviance.argmin()+1, test_deviance.min()+0.02), xycoords='data',
            xytext=(150, 1.0), textcoords='data', arrowprops=dict(arrowstyle="-", connectionstyle="arc"))
ax.annotate('', xy=(800, test_deviance[799]), xycoords='data',
            xytext=(800, est.train_score_[799]), textcoords='data', arrowprops=dict(arrowstyle="-"))
ax.text(810, 0.25, 'train-test gap')

def fmt_params(params):
    return ','.join("{0}={1}".format(key, val) for key, val in params.iteritems())

fig = plt.figure(figsize=(10, 10))
ax = plt.gca()
for params, (test_color, train_color) in [({}, ('#d7191c', '#2c7bb6')),
                                          # ({'min_samples_leaf': 3}, ('#fdae61', '#abd9e9')),
                                          ({'learning_rate': 0.1}, ('#bcbcbc', '#ccebc4')),
                                          ({'learning_rate': 0.1, 'subsample': 0.5}, ('#7A68A6', '#FFB5B8'))]:
    est = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=1, learning_rate=1.0)
    est.set_params(**params)
    est.fit(x_train, y_train)

    ax, test_dev = deviance_plot(est, x_test, y_test, ax=ax, label=fmt_params(params), train_color=train_color, test_color=test_color)

ax.annotate('Higher bias', xy=(900, est.train_score_[899]), xycoords='data',
            xytext=(600, 0.3), textcoords='data', arrowprops=dict(arrowstyle="-", connectionstyle="arc"))
ax.annotate('Lower variance', xy=(900, test_deviance[899]), xycoords='data',
            xytext=(600, 1.0), textcoords='data', arrowprops=dict(arrowstyle="-"))

plt.legend(loc='upper right')

from sklearn.grid_search import GridSearchCV

param_grid = {
    'learning_rate': [0.1, 0.05, 0.02, 0.01],
    'max_depth': [4, 6, 8],
    'min_samples_leaf': [3, 5, 9, 17],
    'subsample': [0.5, 0.7]
    # 'max_features': [1.0, 0.3, 0.1]
}
est = GradientBoostingRegressor(n_estimators=4000)
gs_cv = GridSearchCV(est, param_grid, n_jobs=-1).fit(x_train, y_train) #n_jobs denotes running jobs in parallel
print gs_cv.best_params_

# plt.show()
from sklearn.metrics import log_loss
