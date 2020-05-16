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


def get_data():
    sdr = DataReader_1_0("../data/ch05.npz")
    sdr.ReadData()
    X,Y = sdr.GetWholeTrainSamples()
    x_mean = np.mean(X)
    x_std = np.std(X)
    y_mean = np.mean(Y)
    y_std = np.std(Y)
    ss = StandardScaler()
    X = ss.fit_transform(X)
    Y = ss.fit_transform(Y)

    # test data
    x1 = 15
    x2 = 93
    x = np.array([x1, x2]).reshape(1, 2)
    x_new = NormalizePredicateData(x, x_mean, x_std)

    return X, Y, x_new, y_mean, y_std


def NormalizePredicateData(X_raw, x_mean, x_std):
    X_new = np.zeros(X_raw.shape)
    n = X_raw.shape[1]
    for i in range(n):
        col_i = X_raw[:,i]
        X_new[:,i] = (col_i - x_mean) / x_std
    return X_new


def build_model():
    model = Sequential()
    model.add(Dense(1, activation='linear', input_shape=(2,)))
    model.compile(optimizer='SGD',
                  loss='mse')
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


if __name__ == '__main__':
    X, Y, x_new, y_mean, y_std = get_data()
    x = np.array(X)
    y = np.array(Y)
    print(x.shape)
    print(y.shape)
    print(x)

    model = build_model()
    # patience设置当发现loss没有下降的情况下，经过patience个epoch后停止训练
    early_stopping = EarlyStopping(monitor='loss', patience=100)
    history = model.fit(x, y, epochs=200, batch_size=10, callbacks=[early_stopping])
    w, b = model.layers[0].get_weights()
    print(w)
    print(b)
    plt_loss(history)

    # inference
    z = model.predict(x_new)
    print("z=", z)
    Z_true = z * y_std + y_mean
    print("Z_true=", Z_true)