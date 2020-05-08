# -*- coding: utf-8 -*-#
'''
# Name:         MiniBatchGradientDescent
# Description:  小批量梯度下降
# Author:       super
# Date:         2020/5/8
'''

from HelperClass.NeuralNet_1_0 import *

file_name = "../data/ch04.npz"


def ShowResult(net, dataReader):
    X, Y = dataReader.GetWholeTrainSamples()
    # draw sample data
    plt.plot(X, Y, "b.")
    # draw predication data
    PX = np.linspace(0, 1, 5).reshape(5, 1)
    PZ = net.inference(PX)
    plt.plot(PX, PZ, "r")
    plt.title("Air Conditioner Power")
    plt.xlabel("Number of Servers(K)")
    plt.ylabel("Power of Air Conditioner(KW)")
    plt.show()


if __name__ == '__main__':
    sdr = DataReader_1_0(file_name)
    sdr.ReadData()
    hp = HyperParameters_1_0(1, 1, eta=0.2, max_epoch=100, batch_size=16, eps=0.005)
    net = NeuralNet_1_0(hp)
    net.train(sdr)

    ShowResult(net, sdr)