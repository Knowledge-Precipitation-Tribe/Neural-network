# -*- coding: utf-8 -*-#
'''
# Name:         NN_Complex
# Description:  
# Author:       super
# Date:         2020/5/24
'''

import numpy as np
import matplotlib.pyplot as plt

from HelperClass2.NeuralNet_2_0 import *

train_data_name = "../data/ch09.train.npz"
test_data_name = "../data/ch09.test.npz"

def ShowResult(net, dataReader, title):
    # draw train data
    X,Y = dataReader.XTrain, dataReader.YTrain
    plt.plot(X[:,0], Y[:,0], '.', c='b')
    # create and draw visualized validation data
    TX = np.linspace(0,1,100).reshape(100,1)
    TY = net.inference(TX)
    plt.plot(TX, TY, 'x', c='r')
    plt.title(title)
    plt.show()
#end def

if __name__ == '__main__':
    dataReader = DataReader_2_0(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.GenerateValidationSet()

    n_input, n_hidden, n_output = 1, 3, 1
    eta, batch_size, max_epoch = 0.5, 10, 10000
    eps = 0.001

    hp = HyperParameters_2_0(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.Fitting, InitialMethod.Xavier)
    net = NeuralNet_2_0(hp, "complex_131")

    #加载训练好的权重
    #net.LoadResult()
    net.train(dataReader, 50, True)
    net.ShowTrainingHistory()
    ShowResult(net, dataReader, hp.toString())