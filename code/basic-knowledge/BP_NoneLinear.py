# -*- coding: utf-8 -*-#
'''
# Name:         BP_NoneLinear
# Description:  BP_NoneLinear 非线性反向传播
# Author:       super
# Date:         2020/5/6
'''

import numpy as np
import matplotlib.pyplot as plt

def draw_fun(X,Y):
    x = np.linspace(1.2,10)
    a = x*x
    b = np.log(a)
    c = np.sqrt(b)
    # 正常计算得到的函数曲线
    plt.plot(x,c)

    # 绘制计算的每一步
    plt.plot(X,Y,'x')
    plt.show()

def forward(x):
    a = x * x
    b = np.log(a)
    c = np.sqrt(b)
    return a, b, c

def backward(x, a, b, c, y):
    loss = c - y
    delta_c = loss
    delta_b = delta_c * 2 * np.sqrt(b)
    delta_a = delta_b * a
    delta_x = delta_a / 2 / x
    return loss, delta_x, delta_a, delta_b, delta_c


def update(x, delta_x):
    x = x - delta_x
    if x < 1:
        x = 1.1
    return x


if __name__ == '__main__':
    print("how to play: 1) input x, 2) calculate c, 3) input target number but not faraway from c")
    print("input x as initial number(1.2,10), you can try 1.3:")
    line = input()
    # 输入一个初始值x，x在1.2-10之间
    x = float(line)

    a, b, c = forward(x)
    print("c=%f" % c)
    print("input y as target number(0.5,2), you can try 1.8:")
    line = input()
    # 输入一个你想得到的数字y, 在0.5-2之间，输入的y与计算的c形成误差
    y = float(line)

    error = 1e-3

    X, Y = [], []

    for i in range(20):
        # forward
        print("forward...")
        a, b, c = forward(x)
        print("x=%f,a=%f,b=%f,c=%f" % (x, a, b, c))
        X.append(x)
        Y.append(c)
        # backward
        print("backward...")
        loss, delta_x, delta_a, delta_b, delta_c = backward(x, a, b, c, y)
        if abs(loss) < error:
            print("done!")
            break
        # update x
        x = update(x, delta_x)
        print("delta_c=%f, delta_b=%f, delta_a=%f, delta_x=%f\n" % (delta_c, delta_b, delta_a, delta_x))

    draw_fun(X, Y)