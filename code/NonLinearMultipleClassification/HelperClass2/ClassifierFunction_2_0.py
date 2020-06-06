# -*- coding: utf-8 -*-#
'''
# Name:         ClassifierFunction_2_0
# Description:  分类用到的函数
# Author:       super
# Date:         2020/5/24
'''

import numpy as np

class CClassifier(object):
    def forward(self, z):
        pass

# equal to sigmoid but it is used as classification function
class Logistic(CClassifier):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a

class Softmax(CClassifier):
    def forward(self, z):
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift_z)
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return a