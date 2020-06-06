# -*- coding: utf-8 -*-#
'''
# Name:         Evaluation
# Description:  回归常用评价指标
# Author:       super
# Date:         2020/5/14
'''

import numpy as np
from sklearn import metrics


def MSE(y_true, y_pred):
    return np.average((y_true - y_pred) ** 2, axis=0)
    # return np.mean((y_pred - y_true) * (y_pred - y_true))


def RMSE(y_true, y_pred):
    return np.sqrt(np.average((y_true - y_pred) ** 2, axis=0))


def MAE(y_true, y_pred):
    return np.average(np.abs(y_pred - y_true), axis=0)


def MAPE(y_true, y_pred):
    return np.average(np.abs((y_pred - y_true) / y_true), axis=0) * 100


def SMAPE(y_true, y_pred):
    return 2.0 * np.average(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)), axis=0) * 100


if __name__ == '__main__':
    y_pred = np.array([1.0, 3.3, 2.0, 7.9, 5.5, 6.4, 2.0])
    y_true = np.array([2.0, 3.0, 2.5, 1.0, 4.0, 3.2, 3.0])
    print(metrics.mean_squared_error(y_true=y_true, y_pred=y_pred))
    print(MSE(y_true=y_true, y_pred=y_pred))

    print(np.sqrt(metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)))
    print(RMSE(y_true=y_true, y_pred=y_pred))

    print(metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred))
    print(MAE(y_true=y_true, y_pred=y_pred))
