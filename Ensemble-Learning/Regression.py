# -*- coding: utf-8 -*-#
'''
# Name:         Regression
# Description:  集成学习用于回归，将多个个体学习器的结果最后做一个平均用于输出
# Author:       super
# Date:         2020/6/5
'''
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler


def load_data():
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    x = np.vstack((x_train,x_test))
    y = np.concatenate((y_train, y_test))
    x = StandardScaler().fit_transform(x)
    y = StandardScaler().fit_transform(y.reshape(-1, 1))
    x_train = x[1:401, :]
    x_test = x[401:, :]
    y_train = y[1:401, :]
    y_test = y[401:, :]
    return (x_train, y_train), (x_test, y_test)


def draw_train_history(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(13,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam',
                  loss='mean_squared_error')
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()
    early_stopping = EarlyStopping(monitor='loss', patience=10)
    history = model.fit(x_train, y_train,
                        epochs=500,
                        batch_size=64,
                        validation_split=0.2,
                        callbacks=[early_stopping])
    draw_train_history(history)
    model.save("regression.h5")

    loss = model.evaluate(x_test, y_test, batch_size=64)
    print("test loss: {}".format(loss))