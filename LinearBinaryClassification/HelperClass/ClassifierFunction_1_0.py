# -*- coding: utf-8 -*-#
'''
# Name:         ClassifierFunction
# Description:  分类激活函数
# Author:       super
# Date:         2020/5/16
'''

import numpy as np

# Logisitc激活函数
class Logistic(object):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a

# Tanh激活函数
class Tanh(object):
    def forward(self, z):
        a = 2.0 / (1.0 + np.exp(-2*z)) - 1.0
        return a