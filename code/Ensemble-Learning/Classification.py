# -*- coding: utf-8 -*-#
'''
# Name:         Classification
# Description:  
# Author:       super
# Date:         2020/6/5
'''

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    y_train = OneHotEncoder().fit_transform(y_train.reshape(-1,1))
    y_test = OneHotEncoder().fit_transform(y_test.reshape(-1,1))
    return (x_train, y_train), (x_test, y_test)


def build_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(784, )))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
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
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()
    history = model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=64,
                        validation_split=0.3)
    draw_train_history(history)
    model.save("classification.h5")

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    weights = model.get_weights()
    print("weights: ", weights)