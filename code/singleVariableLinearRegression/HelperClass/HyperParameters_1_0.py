# -*- coding: utf-8 -*-#
'''
# Name:         HyperParameters_1_0
# Description:  神经网络训练中的超参数
# Author:       super
# Date:         2020/5/8
'''

class HyperParameters_1_0(object):
    def __init__(self, input_size, output_size, eta=0.1, max_epoch=1000, batch_size=5, eps=0.1):
        '''
        神经网络中的一些超参数
        :param input_size:  输入数据的大小
        :param output_size: 输出数据的大小
        :param eta: 学习率
        :param max_epoch: 训练多少轮
        :param batch_size: 训练使用数据的大小
        :param eps: 控制训练精度到达多少
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.eta = eta
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.eps = eps

    def toString(self):
        title = str.format("bz:{0},eta:{1}", self.batch_size, self.eta)
        return title