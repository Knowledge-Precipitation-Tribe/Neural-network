# -*- coding: utf-8 -*-#
'''
# Name:         SingleGradientDescent
# Description:  
# Author:       super
# Date:         2020/5/8
'''

from HelperClass.NeuralNet_1_0 import *

file_name = "../data/ch04.npz"

if __name__ == '__main__':
    sdr = DataReader_1_0(file_name)
    sdr.ReadData()
    hp = HyperParameters_1_0(1, 1, eta=0.1, max_epoch=100, batch_size=1, eps = 0.02)
    net = NeuralNet_1_0(hp)
    net.train(sdr)