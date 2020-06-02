# -*- coding: utf-8 -*-#
'''
# Name:         dnn_regression-keras
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
    train_file = "../data/ch09.train.npz"
    test_file = "../data/ch09.test.npz"

    dataReader = DataReader_2_0(train_file, test_file)
    dataReader.ReadData()
    # dr.NormalizeX()
    # dr.NormalizeY(YNormalizationMethod.Regression)
    dataReader.Shuffle()
    dataReader.GenerateValidationSet()

    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    x_test, y_test = dataReader.XTest, dataReader.YTest
    x_val, y_val = dataReader.XDev, dataReader.YDev

    return x_train, y_train, x_test, y_test, x_val, y_val

def build_model():
    model = Sequential()
    model.add(Dense(4, activation='sigmoid', input_shape=(1, )))
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
    x_train, y_train, x_test, y_test, x_val, y_val = load_data()
    # print(x_train.shape)
    # print(x_test.shape)
    # print(x_val.shape)

    model = build_model()
    history = model.fit(x_train, y_train, epochs=50, batch_size=10, validation_data=(x_val, y_val))
    draw_train_history(history)

    loss = model.evaluate(x_test, y_test)
    print("test loss: {}".format(loss))

    weights = model.get_weights()
    print("weights: ", weights)