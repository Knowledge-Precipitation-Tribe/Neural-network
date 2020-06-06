# -*- coding: utf-8 -*-#
'''
# Name:         lDataNormalization
# Description:  
# Author:       super
# Date:         2020/5/13
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from HelperClass.NeuralNet_1_1 import *

file_name = "../data/ch05.npz"

def ShowResult(net, reader):
    # draw example points
    X,Y = reader.GetWholeTrainSamples()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:,0],X[:,1],Y)
    # draw fitting surface
    # numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
    # Return evenly spaced numbers over a specified interval.
    # 返回指定间隔内的均匀间隔的数字。
    # >>> np.linspace(2.0, 3.0, num=5)
    # array([2.  , 2.25, 2.5 , 2.75, 3.  ])
    # >>> np.linspace(2.0, 3.0, num=5, endpoint=False)
    # array([2. ,  2.2,  2.4,  2.6,  2.8])
    #
    # retstep : bool, optional
    # If True, return (samples, step), where step is the spacing between samples.
    # >>> np.linspace(2.0, 3.0, num=5, retstep=True)
    # (array([2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)
    p = np.linspace(0,1)
    q = np.linspace(0,1)
    # Return coordinate matrices from coordinate vectors.
    # 从坐标向量返回坐标矩阵。
    # nx, ny = (3,2)
    # x = np.linspace(0, 1, nx)
    # y = np.linspace(0, 1, ny)
    # print(x)
    # [0.  0.5 1. ]
    # print(y)
    # [0. 1.]
    # xv, yv = np.meshgrid(x, y)
    # print(xv)
    # [[0.  0.5 1. ]
    #  [0.  0.5 1. ]]
    # print(yv)
    # [[0. 0. 0.]
    #  [1. 1. 1.]]
    #
    # yv1, xv1 = np.meshgrid(y, x)
    # print(xv1)
    # [[0.  0. ]
    #  [0.5 0.5]
    #  [1.  1. ]]
    # print(yv1)
    # [[0. 1.]
    #  [0. 1.]
    #  [0. 1.]]
    P,Q = np.meshgrid(p,q)
    R = np.hstack((P.ravel().reshape(2500,1), Q.ravel().reshape(2500,1)))
    Z = net.inference(R)
    Z = Z.reshape(50,50)
    ax.plot_surface(P,Q,Z, cmap='rainbow')
    plt.show()


if __name__ == '__main__':
    # data
    reader = DataReader_1_1(file_name)
    reader.ReadData()
    reader.NormalizeX()
    # use this setting for the first time
    hp = HyperParameters_1_0(2, 1, eta=0.1, max_epoch=10, batch_size=1, eps = 1e-5)
    # use this setting when you want to train more loops
    #hp = HyperParameters_1_0(2, 1, eta=0.01, max_epoch=500, batch_size=10, eps = 1e-5)
    net = NeuralNet_1_1(hp)
    net.train(reader, checkpoint=0.1)
    # inference
    x1 = 15
    x2 = 93
    x = np.array([x1,x2]).reshape(1,2)
    print("z=", net.inference(x))

    ShowResult(net, reader)