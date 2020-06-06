# -*- coding: utf-8 -*-#
'''
# Name:         NormalizePredicateData
# Description:  将测试数据也进行归一化操作
# Author:       super
# Date:         2020/5/13
'''

import numpy as np

from HelperClass.NeuralNet_1_1 import *

file_name = "../data/ch05.npz"

if __name__ == '__main__':
    # data
    reader = DataReader_1_1(file_name)
    reader.ReadData()
    reader.NormalizeX()
    # net
    hp = HyperParameters_1_0(2, 1, eta=0.01, max_epoch=100, batch_size=10, eps = 1e-5)
    net = NeuralNet_1_1(hp)
    net.train(reader, checkpoint=0.1)
    # inference
    x1 = 15
    x2 = 93
    x = np.array([x1,x2]).reshape(1,2)
    x_new = reader.NormalizePredicateData(x)
    z = net.inference(x_new)
    print("Z=", z)