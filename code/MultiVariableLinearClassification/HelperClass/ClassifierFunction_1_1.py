# -*- coding: utf-8 -*-#
'''
# Name:         ClassifierFunction_1_1
# Description:  add softmax
# Author:       super
# Date:         2020/5/22
'''

import numpy as np

class Logistic(object):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a

class Softmax(object):
    def forward(self, z):
        # 当x很大时，np.exp很容易造成溢出，所以进行以下操作
        shift_z = z - np.max(z, axis=1, keepdims=True)
        shift_z = z
        exp_z = np.exp(shift_z)
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return a