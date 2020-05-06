# -*- coding: utf-8 -*-#
'''
# Name:         BP_Linear
# Description:  BP_Linear 线性反向传播
# Author:       super
# Date:         2020/5/6
'''

import numpy as np

def target_function(w, b):
    x = 2 * w + 3 * b
    y = 2 * b + 1
    z = x * y
    return x, y, z

def back_propagation_for_w(w, b, t):
    '''
    反向传播求解w
    :param w: 权重w
    :param b: 权重b
    :param t: 目标值t
    :return:
    '''
    print("\nback_propagation_for_w ----- \n")
    error = 1e-5
    count = 1
    while(True):
        x, y ,z = target_function(w, b)
        delta_z = z - t
        print("w=%f,b=%f,z=%f,delta_z=%f" % (w, b, z, delta_z))
        if abs(delta_z) < error:
            break
        # 偏z偏x=y
        # 偏x偏w=2
        partial_w = y * 2
        delta_w = delta_z / partial_w
        w = w - delta_w
        count = count + 1
    print("done!\ntotal iteration times = %d" % count)
    print("final w = %f" % w)

def back_propagation_for_b(w, b, t):
    '''
    反向传播求解b
    :param w: 权重w
    :param b: 权重b
    :param t: 目标值t
    :return:
    '''
    print("\nback_propagation_for_b ----- \n")
    error = 1e-5
    count = 1
    while(True):
        x, y ,z = target_function(w, b)
        delta_z = z - t
        print("w=%f,b=%f,z=%f,delta_z=%f" % (w, b, z, delta_z))
        if abs(delta_z) < error:
            break
        # 偏z偏x=y
        # 偏x偏b=3
        # 偏z偏y=x
        # 偏y偏b=2
        partial_b = 2 * x + 3 * y
        delta_b = delta_z / partial_b
        b = b - delta_b
        count = count + 1
    print("done!\ntotal iteration times = %d" % count)
    print("final b = %f" % b)

def back_propagation_for_wb(w, b, t):
    '''
    反向传播求解wb
    :param w: 权重w
    :param b: 权重b
    :param t: 目标值t
    :return:
    '''
    print("\nback_propagation_for_wb ----- \n")
    error = 1e-5
    count = 1
    while(True):
        x, y ,z = target_function(w, b)
        delta_z = z - t
        print("w=%f,b=%f,z=%f,delta_z=%f" % (w, b, z, delta_z))
        if abs(delta_z) < error:
            break
        # 偏z偏x=y
        # 偏x偏b=3
        # 偏z偏y=x
        # 偏y偏b=2
        # 偏z偏x=y
        # 偏x偏w=2
        partial_b = 2 * x + 3 * y
        partial_w = 2 * y
        # 同时求解w和b，将误差均分到w和b上
        delta_b = delta_z / partial_b / 2
        delta_w = delta_z / partial_w / 2
        b = b - delta_b
        w = w - delta_w
        count = count + 1
    print("done!\ntotal iteration times = %d" % count)
    print("final b = %f" % b)
    print("final w = %f" % w)

if __name__ == '__main__':
    w = 3
    b = 4
    t = 150
    back_propagation_for_w(w, b, t)
    back_propagation_for_b(w, b, t)
    back_propagation_for_wb(w, b, t)