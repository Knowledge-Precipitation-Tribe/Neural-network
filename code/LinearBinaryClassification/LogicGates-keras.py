# -*- coding: utf-8 -*-#
'''
# Name:         LogicGates-keras
# Description:  
# Author:       super
# Date:         2020/5/22
'''

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

from HelperClass.NeuralNet_1_2 import *
from HelperClass.Visualizer_1_0 import *


class LogicDataReader(DataReader_1_1):
    def __init__(self):
        pass

    def Read_Logic_NOT_Data(self):
        X = np.array([0, 1]).reshape(2, 1)
        Y = np.array([1, 0]).reshape(2, 1)
        self.XTrain = self.XRaw = X
        self.YTrain = self.YRaw = Y
        self.num_train = self.XRaw.shape[0]

    def Read_Logic_AND_Data(self):
        X = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        Y = np.array([0, 0, 0, 1]).reshape(4, 1)
        self.XTrain = self.XRaw = X
        self.YTrain = self.YRaw = Y
        self.num_train = self.XRaw.shape[0]

    def Read_Logic_NAND_Data(self):
        X = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        Y = np.array([1, 1, 1, 0]).reshape(4, 1)
        self.XTrain = self.XRaw = X
        self.YTrain = self.YRaw = Y
        self.num_train = self.XRaw.shape[0]

    def Read_Logic_OR_Data(self):
        X = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        Y = np.array([0, 1, 1, 1]).reshape(4, 1)
        self.XTrain = self.XRaw = X
        self.YTrain = self.YRaw = Y
        self.num_train = self.XRaw.shape[0]

    def Read_Logic_NOR_Data(self):
        X = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        Y = np.array([1, 0, 0, 0]).reshape(4, 1)
        self.XTrain = self.XRaw = X
        self.YTrain = self.YRaw = Y
        self.num_train = self.XRaw.shape[0]


def build_model():
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_shape=(2,)))
    model.compile(optimizer='SGD', loss='binary_crossentropy')
    return model


def draw_source_data(reader, title, show=False):
    fig = plt.figure(figsize=(5, 5))
    plt.grid()
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.title(title)
    X, Y = reader.GetWholeTrainSamples()
    if title == "Logic NOT operator":
        DrawTwoCategoryPoints(X[:, 0], np.zeros_like(X[:, 0]), Y[:, 0], title=title, show=show)
    else:
        DrawTwoCategoryPoints(X[:, 0], X[:, 1], Y[:, 0], title=title, show=show)


def draw_split_line(w, b):
    x = np.array([-0.1, 1.1])
    old_w = w
    w = -w[0,0]/old_w[1,0]
    b = -b[0]/old_w[1,0]
    y = w * x + b
    plt.plot(x, y)


if __name__ == '__main__':
    reader = LogicDataReader()
    reader.Read_Logic_AND_Data()
    x, y = reader.XTrain, reader.YTrain
    print("x", x)
    print("y", y)
    model = build_model()
    model.fit(x, y, epochs=1000, batch_size=1)
    # 获得权重
    w, b = model.layers[0].get_weights()
    print(w)
    print(b)

    draw_source_data(reader, "Logic AND operator")
    draw_split_line(w, b)
    plt.show()