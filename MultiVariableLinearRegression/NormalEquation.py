# -*- coding: utf-8 -*-#
'''
# Name:         NormalEquation
# Description:  
# Author:       super
# Date:         2020/5/13
'''

import numpy as np

from HelperClass.DataReader_1_1 import *

file_name = "../data/ch05.npz"

if __name__ == '__main__':
    reader = DataReader_1_1(file_name)
    reader.ReadData()
    X,Y = reader.GetWholeTrainSamples()
    num_example = X.shape[0]
    one = np.ones((num_example,1))
    # Stack 1-D arrays as columns into a 2-D array.
    # >>> a = np.array((1,2,3))
    # >>> b = np.array((2,3,4))
    # >>> np.column_stack((a,b))
    # array([[1, 2],
    #        [2, 3],
    #        [3, 4]])
    x = np.column_stack((one, (X[0:num_example,:])))

    a = np.dot(x.T, x)
    # need to convert to matrix,
    # because np.linalg.inv only works on matrix instead of array
    b = np.asmatrix(a)
    c = np.linalg.inv(b)
    d = np.dot(c, x.T)
    e = np.dot(d, Y)
    #print(e)
    b=e[0,0]
    w1=e[1,0]
    w2=e[2,0]
    print("w1=", w1)
    print("w2=", w2)
    print("b=", b)
    # inference
    z = w1 * 15 + w2 * 93 + b
    print("z=",z)