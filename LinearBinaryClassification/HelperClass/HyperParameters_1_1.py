# -*- coding: utf-8 -*-#
'''
# Name:         HyperParameters
# Description:  设置网络超参数
# Author:       super
# Date:         2020/5/16
'''
from HelperClass.EnumDef_1_0 import *

class HyperParameters_1_1(object):
    def __init__(self, input_size, output_size, eta=0.1, max_epoch=1000, batch_size=5, eps=0.1, net_type=NetType.Fitting):
        self.input_size = input_size
        self.output_size = output_size
        self.eta = eta
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.eps = eps
        self.net_type = net_type

    def toString(self):
        title = str.format("bz:{0},eta:{1}", self.batch_size, self.eta)
        return title
