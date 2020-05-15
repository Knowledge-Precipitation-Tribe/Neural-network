# -*- coding: utf-8 -*-#
'''
# Name:         NeuralNetwork-keras
# Description:  多入单处层神经网络
# Author:       super
# Date:         2020/5/15
'''

from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler

from HelperClass.DataReader_1_0 import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_data():
    sdr = DataReader_1_0("../data/ch05.npz")
    sdr.ReadData()
    X,Y = sdr.GetWholeTrainSamples()
    ss = StandardScaler()
    X = ss.fit_transform(X)
    Y = ss.fit_transform(Y)
    return X, Y


def build_model():
    model = Sequential()
    model.add(Dense(1, activation='linear', input_shape=(2,)))
    model.compile(optimizer='SGD',
                  loss='mse')
    return model


def plt_3d(model, X, Y):
    # draw example points
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:,0],X[:,1],Y)
    p = np.linspace(-2,1)
    q = np.linspace(-2,2)
    P,Q = np.meshgrid(p,q)
    R = np.hstack((P.ravel().reshape(2500,1), Q.ravel().reshape(2500,1)))
    Z = model.predict(R)
    Z = Z.reshape(50,50)
    ax.plot_surface(P,Q,Z, cmap='rainbow')
    plt.show()


def plt_loss(history):
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.title('Training  loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    X, Y = get_data()
    x = np.array(X)
    y = np.array(Y)
    print(x.shape)
    print(y.shape)
    print(x)

    model = build_model()
    # patience设置当发现loss没有下降的情况下，经过patience个epoch后停止训练
    early_stopping = EarlyStopping(monitor='loss', patience=100)
    history = model.fit(x, y, epochs=100, batch_size=32, callbacks=[early_stopping])
    w, b = model.layers[0].get_weights()
    print(w, b)
    plt_3d(model, x, y)
    plt_loss(history)