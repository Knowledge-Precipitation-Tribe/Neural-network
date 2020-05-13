# -*- coding: utf-8 -*-#
'''
# Name:         NeuralNetwork
# Description:  
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
    # net
    hp = HyperParameters_1_0(2, 1, eta=0.1, max_epoch=10, batch_size=1, eps = 1e-5)
    net = NeuralNet_1_1(hp)
    net.train(reader, checkpoint=0.1)
    # inference
    x1 = 15
    x2 = 93
    x = np.array([x1,x2]).reshape(1,2)
    print(net.inference(x))