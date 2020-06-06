# -*- coding: utf-8 -*-#
'''
# Name:         NN_Sin-keras
# Description:  
# Author:       super
# Date:         2020/5/24
'''

from HelperClass2.DataReader_2_0 import *

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
from contextlib import redirect_stdout

import os

def load_data():
    train_data_name = "../data/ch08.train.npz"
    test_data_name = "../data/ch08.test.npz"

    dataReader = DataReader_2_0(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.GenerateValidationSet()

    x_train, y_train, x_val, y_val = dataReader.XTrain, dataReader.YTrain, dataReader.XDev, dataReader.YDev
    x_test, y_test = dataReader.XTest, dataReader.YTest
    return x_train, y_train, x_val, y_val, x_test, y_test


def build_model():
    model = Sequential()
    model.add(Dense(2, activation='sigmoid', input_shape=(1,)))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='Adam',
                  loss='mse')
    return model


def draw_train_history(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    model_path = "nn_sin_keras/nn_sin.h5"
    model_weights_path = "nn_sin_keras/nn_sin_weights.h5"

    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = build_model()
        early_stopping = EarlyStopping(monitor='loss', patience=100)
        history = model.fit(x_train, y_train,
                            epochs=5000,
                            batch_size=32,
                            callbacks=[early_stopping],
                            validation_data=(x_val, y_val))
        draw_train_history(history)
        loss= model.evaluate(x_test, y_test, batch_size=32)
        print("test loss: {}".format(loss))
        model.save(model_path)
        model.save_weights(model_weights_path)

    model_summary_path = "nn_sin_keras/nn_sin_summary.txt"
    with open(model_summary_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()