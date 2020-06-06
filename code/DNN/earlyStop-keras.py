# -*- coding: utf-8 -*-#
'''
# Name:         earlyStop-keras
# Description:  
# Author:       super
# Date:         2020/6/4
'''

from ExtendedDataReader.MnistImageDataReader import *

from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, BatchNormalization
from keras.regularizers import l2, l1

import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_data():
    dataReader = MnistImageDataReader(mode="vector")
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.NormalizeY(NetType.MultipleClassifier)
    dataReader.GenerateValidationSet(k=20)

    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    x_test, y_test = dataReader.XTest, dataReader.YTest
    x_val, y_val = dataReader.XDev, dataReader.YDev

    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_val = x_val.reshape(x_val.shape[0], 28 * 28)


    return x_train, y_train, x_test, y_test, x_val, y_val

def build_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(784, )))
    model.add(BatchNormalization())
    model.add(Dense(64,
                    activation='relu',
                    kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(32,
                    activation='relu',
                    kernel_regularizer=l1(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#画出训练过程中训练和验证的精度与损失
def draw_train_history(history):
    plt.figure(1)

    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, x_val, y_val = load_data()
    # print(x_train.shape)
    # print(x_test.shape)
    # print(x_val.shape)

    model = build_model()
    early_stopping = EarlyStopping(monitor='loss', patience=8)
    history = model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=64,
                        validation_data=(x_val, y_val),
                        callbacks=[early_stopping])
    draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    weights = model.get_weights()
    print("weights: ", weights)