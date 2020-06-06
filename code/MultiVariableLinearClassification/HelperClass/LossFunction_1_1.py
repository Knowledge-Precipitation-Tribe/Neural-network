# -*- coding: utf-8 -*-#
'''
# Name:         LossFunction_1_1
# Description:  add mse, Crossentropy for multiple classifier
# Author:       super
# Date:         2020/5/22
'''

import numpy as np

from HelperClass.EnumDef_1_0 import *

class LossFunction_1_1(object):
    def __init__(self, net_type):
        self.net_type = net_type
    # end def

    # fcFunc: feed forward calculation
    def CheckLoss(self, A, Y):
        m = Y.shape[0]
        if self.net_type == NetType.Fitting:
            loss = self.MSE(A, Y, m)
        elif self.net_type == NetType.BinaryClassifier:
            loss = self.CE2(A, Y, m)
        elif self.net_type == NetType.MultipleClassifier:
            loss = self.CE3(A, Y, m)
        #end if
        return loss
    # end def

    def MSE(self, A, Y, count):
        p1 = A - Y
        LOSS = np.multiply(p1, p1)
        loss = LOSS.sum()/count/2
        return loss
    # end def

    # for binary classifier
    def CE2(self, A, Y, count):
        p1 = 1 - Y
        p2 = np.log(1 - A)
        p3 = np.log(A)

        p4 = np.multiply(p1 ,p2)
        p5 = np.multiply(Y, p3)

        LOSS = np.sum(-(p4 + p5))  #binary classification
        loss = LOSS / count
        return loss
    # end def

    # for multiple classifier
    def CE3(self, A, Y, count):
        '''
        交叉熵函数：-y*loga
        :param A: 神经网络的计算结果，经过softmax函数计算
        :param Y: 标签值
        :param count:
        :return:
        '''
        p1 = np.log(A)
        p2 =  np.multiply(Y, p1)
        LOSS = np.sum(-p2)
        loss = LOSS / count
        return loss
    # end def