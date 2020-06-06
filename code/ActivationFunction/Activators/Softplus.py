# -*- coding: utf-8 -*-#
'''
# Name:         Softplus
# Description:  
# Author:       super
# Date:         2020/5/24
'''

import numpy as np
import matplotlib.pyplot as plt

class CSoftplus(object):
    def forward(self, z):
        a = np.log(1 + np.exp(z))
        return a

    def backward(self, z, a, delta):
        p = np.exp(z)
        da = p / (1 + p)
        dz = np.multiply(delta, da)
        return da, dz