# -*- coding: utf-8 -*-#
'''
# Name:         NeuralNet-keras
# Description:  
# Author:       super
# Date:         2020/5/15
'''

from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from HelperClass.DataReader_1_0 import *

import matplotlib.pyplot as plt


def get_data():
    sdr = DataReader_1_0("../data/ch04.npz")
    sdr.ReadData()
    X,Y = sdr.GetWholeTrainSamples()
    return X, Y


def build_model():
    model = Sequential()
    model.add(Dense(1, activation='linear', input_dim=1))
    model.compile(optimizer='SGD',
                  loss='mse')
    return model


def plt_data(x, y, model):
    # draw sample data
    plt.plot(x, y, "b.")
    # draw predication data
    PX = np.linspace(0,1,10)
    PZ = model.predict(PX)
    plt.plot(PX, PZ, "r")
    plt.title("Air Conditioner Power")
    plt.xlabel("Number of Servers(K)")
    plt.ylabel("Power of Air Conditioner(KW)")
    plt.show()


if __name__ == '__main__':
    X, Y = get_data()
    x = np.array(X)
    y = np.array(Y)
    print(x.shape)
    print(y.shape)

    model = build_model()
    # patience设置当发现loss没有下降的情况下，经过patience个epoch后停止训练
    early_stopping = EarlyStopping(monitor='loss', patience=100)
    model.fit(x, y, epochs=100, batch_size=10, callbacks=[early_stopping])
    w, b = model.layers[0].get_weights()
    print(w, b)
    plt_data(x, y, model)