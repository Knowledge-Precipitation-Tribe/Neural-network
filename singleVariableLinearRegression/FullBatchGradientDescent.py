# -*- coding: utf-8 -*-#
'''
# Name:         FullBatchGradientDescent
# Description:  全批量梯度下降
# Author:       super
# Date:         2020/5/8
'''

from HelperClass.NeuralNet_1_0 import *

file_name = "../data/ch04.npz"

if __name__ == '__main__':
    sdr = DataReader_1_0(file_name)
    sdr.ReadData()
    # batch_size=1即为全批量
    hp = HyperParameters_1_0(1, 1, eta=0.5, max_epoch=1000, batch_size=-1, eps = 0.02)
    net = NeuralNet_1_0(hp)
    net.train(sdr)