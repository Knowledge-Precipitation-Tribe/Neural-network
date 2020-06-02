# -*- coding: utf-8 -*-#
'''
# Name:         HousePriceRegression-keras
# Description:  
# Author:       super
# Date:         2020/6/2
'''

from HelperClass2.MnistImageDataReader import *

from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_data():
    train_file = "../data/ch14.house.train.npz"
    test_file = "../data/ch14.house.test.npz"

    dataReader = DataReader_2_0(train_file, test_file)
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.NormalizeY(NetType.Fitting)
    dataReader.Shuffle()
    dataReader.GenerateValidationSet(k=10)
    return dataReader


def gen_data(dataReader):
    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    x_test, y_test = dataReader.XTest, dataReader.YTest
    x_val, y_val = dataReader.XDev, dataReader.YDev

    return x_train, y_train, x_test, y_test, x_val, y_val


def showResult(net, dr):
    y_test_result = net.predict(dr.XTest[0:1000, :])
    y_test_real = dr.DeNormalizeY(y_test_result)
    plt.scatter(y_test_real, y_test_real - dr.YTestRaw[0:1000, :], marker='o', label='test data')
    plt.show()


def build_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(13, )))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='Adam',
                  loss='mean_squared_error')
    return model

#画出训练过程中训练和验证的精度与损失
def draw_train_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    dataReader = load_data()
    x_train, y_train, x_test, y_test, x_val, y_val = gen_data(dataReader)

    model = build_model()
    history = model.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x_val, y_val))
    draw_train_history(history)
    showResult(model, dataReader)

    loss = model.evaluate(x_test, y_test)
    print("test loss: {}".format(loss))

    weights = model.get_weights()
    print("weights: ", weights)