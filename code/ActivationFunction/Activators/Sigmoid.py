# -*- coding: utf-8 -*-#
'''
# Name:         Sigmoid
# Description:  具体推导：https://docs.nn.knowledge-precipitation.site/fei-xian-xing-hui-gui/ji-ya-xing-ji-huo-han-shu
# Author:       super
# Date:         2020/5/24
'''

import numpy as np
import matplotlib.pyplot as plt

class CSigmoid(object):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a

    def backward(self, z, a, delta):
        da = np.multiply(a, 1-a)
        dz = np.multiply(delta, da)
        return da, dz