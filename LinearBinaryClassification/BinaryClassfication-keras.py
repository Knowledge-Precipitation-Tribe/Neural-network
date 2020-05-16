# -*- coding: utf-8 -*-#
'''
# Name:         BinaryClassfication-keras
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


if __name__ == '__main__':
    X, Y = get_data()
    x = np.array(X)
    y = np.array(Y)
    print(x.shape)
    print(y.shape)
    print(x)
    # print(y)

    model = build_model()
    early_stopping = EarlyStopping(monitor='loss', patience=100)
    history = model.fit(x, y, epochs=500, batch_size=10, callbacks=[early_stopping])
    w, b = model.layers[0].get_weights()
    print(w)
    print(b)
    plt_loss(history)

    # inference
    x_predicate = np.array([0.58, 0.92, 0.62, 0.55, 0.39, 0.29]).reshape(3, 2)
    a = model.predict(x_predicate)
    print("A=", a)