# -*- coding: utf-8 -*-#
'''
# Name:         BenIdentity
# Description:  
# Author:       super
# Date:         2020/5/24
'''

import numpy as np
import matplotlib.pyplot as plt

class CBenIdentity(object):
    def forward(self, z):
        p1 = np.multiply(z, z)
        p2 = np.sqrt(p1 + 1)
        a = (p2 - 1) / 2 + z
        return a

    def backward(self, z, a, delta):
        da = z / (2 * np.sqrt(z ** 2 + 1)) + 1
        dz = np.multiply(da, delta)
        return da, dz