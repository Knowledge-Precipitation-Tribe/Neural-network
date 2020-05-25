# -*- coding: utf-8 -*-#
'''
# Name:         XorGateClassifier-keras
# Description:  
# Author:       super
# Date:         2020/5/25
'''

from XorGateClassifier import *

from keras.models import Sequential
from keras.layers import Dense

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_data():
    dataReader = XOR_DataReader()
    dataReader.ReadData()
    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    return x_train, y_train


def build_model():
    model = Sequential()
    model.add(Dense(2, activation='sigmoid', input_shape=(2, )))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
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
    plt.legend(['train', 'validation'], loc='upper left')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    x_train, y_train = load_data()

    model = build_model()
    history = model.fit(x_train, y_train, epochs=5000, batch_size=1, validation_data=(x_train, y_train))
    draw_train_history(history)

    loss, accuracy = model.evaluate(x_train, y_train)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    weights = model.get_weights()
    print("weights: ", weights)