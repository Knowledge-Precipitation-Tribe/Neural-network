# -*- coding: utf-8 -*-#
'''
# Name:         GDSingleVariable
# Description:  GDSingleVariable 单变量梯度下降
# Author:       super
# Date:         2020/5/7
'''

import numpy as np
import matplotlib.pyplot as plt

def target_function(x):
    '''
    目标函数
    :param x:
    :return:
    '''
    y = x * x
    return y

def derivative_function(x):
    '''
    目标函数导数
    :param x:
    :return:
    '''
    return 2*x


def draw_function():
    x = np.linspace(-1.2, 1.2)
    y = target_function(x)
    plt.plot(x, y)


def draw_gd(X, Y):
    plt.plot(X, Y)


if __name__ == '__main__':
    x = 1.2
    eta = 0.3
    error = 1e-3
    X = []
    X.append(x)
    Y = []
    y = target_function(x)
    Y.append(y)
    while y > error:
        x = x - eta * derivative_function(x)
        X.append(x)
        y = target_function(x)
        Y.append(y)
        print("x=%f, y=%f" % (x, y))

    draw_function()
    draw_gd(X,Y)
    plt.show()