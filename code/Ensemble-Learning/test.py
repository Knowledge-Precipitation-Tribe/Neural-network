# -*- coding: utf-8 -*-#
'''
# Name:         test
# Description:  
# Author:       super
# Date:         2020/6/6
'''
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve


def load_data():
    x, y = load_boston(return_X_y=True)
    x = StandardScaler().fit_transform(x)
    y = StandardScaler().fit_transform(y.reshape(-1, 1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)
    return (x_train, y_train), (x_test, y_test)

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(13, )))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam',
                  loss='mean_squared_error')
    return model


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    # mean为求矩阵的均值，std为求矩阵的标准差
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    # 在中位数曲线的上下描绘出标准差的范围，以带颜色的透明度为0.1的区域显示
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # 画出两条中位数曲线
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    # 绘制图例，参数loc表示图例放置的位置，best表示自适应
    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()

    model = LinearRegression(n_jobs=-1)
    model.fit(x_train, y_train)
    plot_learning_curve(model, "linear", x_train, y_train)

    print(model.score(x_test, y_test))