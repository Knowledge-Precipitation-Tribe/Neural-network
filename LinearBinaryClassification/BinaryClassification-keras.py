# -*- coding: utf-8 -*-#
'''
# Name:         BinaryClassification-keras
# Description:  keras实现线性二分类
# Author:       super
# Date:         2020/5/16
'''

from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler

from HelperClass.DataReader_1_1 import *

import matplotlib.pyplot as plt

def get_data():
    sdr = DataReader_1_1("../data/ch06.npz")
    sdr.ReadData()
    X, Y = sdr.GetWholeTrainSamples()
    ss = StandardScaler()
    X = ss.fit_transform(X)
    # Y = ss.fit_transform(Y)
    return X, Y


def build_model():
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_shape=(2, )))
    model.compile(optimizer='SGD', loss='binary_crossentropy')
    return model


def plt_loss(history):
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.title('Training  loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plt_data(X, Y):
    DrawTwoCategoryPoints(X[:, 0], X[:, 1], Y[:, 0], show=False)

def plt_split_line(w, b):
    b12 = -b[0]/w[1,0]
    w12 = -w[0,0]/w[1,0]
    print("w12=", w12)
    print("b12=", b12)
    x = np.linspace(-2,2,10)
    y = w12 * x + b12
    plt.plot(x,y)
    plt.axis([-2,2,-2,2])
    plt.show()


def plt_predicate_data(net, threshold=0.5):
    x = np.array([0.58,0.92,0.62,0.55,0.39,0.29]).reshape(3,2)
    a = net.predict(x)
    print("A=", a)
    DrawTwoCategoryPoints(x[:,0], x[:,1], a[:,0], show=False, isPredicate=True)

def DrawTwoCategoryPoints(X1, X2, Y, xlabel="x1", ylabel="x2", title=None, show=False, isPredicate=False):
    colors = ['b', 'r']
    shapes = ['s', 'x']
    assert(X1.shape[0] == X2.shape[0] == Y.shape[0])
    count = X1.shape[0]
    for i in range(count):
        j = (int)(round(Y[i]))
        if j < 0:
            j = 0
        if isPredicate:
            plt.scatter(X1[i], X2[i], color=colors[j], marker='^', s=200, zorder=10)
        else:
            plt.scatter(X1[i], X2[i], color=colors[j], marker=shapes[j], zorder=10)
    # end for
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


if __name__ == '__main__':
    X, Y = get_data()
    x = np.array(X)
    y = np.array(Y)
    print(x.shape)
    print(y.shape)
    print(x)

    model = build_model()
    early_stopping = EarlyStopping(monitor='loss', patience=100)
    history = model.fit(x, y, epochs=1000, batch_size=10, callbacks=[early_stopping])
    w, b = model.layers[0].get_weights()
    print(w)
    print(b)
    # # inference
    # x_predicate = np.array([0.58, 0.92, 0.62, 0.55, 0.39, 0.29]).reshape(3, 2)
    # a = model.predict(x_predicate)
    # print("A=", a)

    # 在这里直接进行预测
    plt_data(X, Y)
    # plt.show()
    plt_predicate_data(model)
    plt_split_line(w, b)
    plt.show()
    plt_loss(history)