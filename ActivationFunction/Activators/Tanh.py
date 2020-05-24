# -*- coding: utf-8 -*-#
'''
# Name:         Tanh
# Description:  具体推导：https://docs.nn.knowledge-precipitation.site/fei-xian-xing-hui-gui/ji-ya-xing-ji-huo-han-shu
# Author:       super
# Date:         2020/5/24
'''

import numpy as np
import matplotlib.pyplot as plt

class CTanh(object):
    def forward(self, z):
        a = 2.0 / (1.0 + np.exp(-2*z)) - 1.0
        return a

    def backward(self, z, a, delta):
        da = 1 - np.multiply(a, a)
        dz = np.multiply(delta, da)
        return da, dz