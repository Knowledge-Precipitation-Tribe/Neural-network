# -*- coding: utf-8 -*-#
'''
# Name:         Relu
# Description:  
# Author:       super
# Date:         2020/5/24
'''

import numpy as np
import matplotlib.pyplot as plt

class CRelu(object):
    def forward(self, z):
        # 把所有大于0的单元填充为1,反向是不用再次计算
        self.mem = np.zeros(z.shape)
        self.mem[z>0] = 1
        a = np.maximum(z, 0)
        return a

    def backward(self, z, a, delta):
        da = np.array([1 if x > 0 else 0 for x in a])
        dz = self.mem * delta
        return da, dz