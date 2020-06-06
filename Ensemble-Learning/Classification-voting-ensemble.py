# -*- coding: utf-8 -*-#
'''
# Name:         Classification-voting-ensemble
# Description:  
# Author:       super
# Date:         2020/6/5
'''

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

from pathlib import Path

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    y_train = LabelEncoder().fit_transform(y_train.reshape(-1,1))
    y_test = LabelEncoder().fit_transform(y_test.reshape(-1,1))
    return (x_train, y_train), (x_test, y_test)


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


def build_model(hidden_units):
    model = Sequential()
    for index, unit in enumerate(hidden_units):
        if index == 0:
            model.add(Dense(unit, activation='relu', input_shape=(784, )))
        else:
            model.add(Dense(unit, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_model1():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(784, )))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_model2():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(784, )))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_model3():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(784, )))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()
    print(y_test.shape)

    model1 = KerasClassifier(build_fn=build_model1, epochs=2, batch_size=64)
    model1._estimator_type = "classifier"
    model2 = KerasClassifier(build_fn=build_model2, epochs=2, batch_size=64)
    model2._estimator_type = "classifier"
    model3 = KerasClassifier(build_fn=build_model3, epochs=2, batch_size=64)
    model3._estimator_type = "classifier"

    # if ‘hard’, uses predicted class labels for majority rule voting.
    # if ‘soft’, predicts the class label based on the argmax of the
    # sums of the predicted probabilities,
    # which is recommended for an ensemble of well-calibrated classifiers.
    cls = VotingClassifier(estimators=(['model1', model1],
                                       ['model2', model2],
                                       ['model3', model3]),
                           voting='hard')
    cls.fit(x_train, y_train)
    
    print("score: ", cls.score(x_test, y_test))