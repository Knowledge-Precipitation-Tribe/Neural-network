# -*- coding: utf-8 -*-#
'''
# Name:         dnn_multiClassification-keras
# Description:  
# Author:       super
# Date:         2020/6/2
'''


from MiniFramework.DataReader_2_0 import *

from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_data():
    train_file = "../data/ch11.train.npz"
    test_file = "../data/ch11.test.npz"

    dataReader = DataReader_2_0(train_file, test_file)
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.NormalizeY(NetType.MultipleClassifier, base=1)
    dataReader.Shuffle()
    dataReader.GenerateValidationSet()

    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    x_test, y_test = dataReader.XTest, dataReader.YTest
    x_val, y_val = dataReader.XDev, dataReader.YDev

    return x_train, y_train, x_test, y_test, x_val, y_val

def build_model():
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(2, )))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))
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

    model = build_model()
    history = model.fit(x_train, y_train, epochs=100, batch_size=10, validation_data=(x_val, y_val))
    draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    weights = model.get_weights()
    print("weights: ", weights)