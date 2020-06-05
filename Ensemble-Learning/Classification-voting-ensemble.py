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

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import VotingClassifier

from pathlib import Path

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    y_train = OneHotEncoder().fit_transform(y_train.reshape(-1,1))
    y_test = OneHotEncoder().fit_transform(y_test.reshape(-1,1))
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

    model1 = KerasClassifier(build_fn=build_model1, epochs=20, batch_size=64)
    model2 = KerasClassifier(build_fn=build_model2, epochs=20, batch_size=64)
    model3 = KerasClassifier(build_fn=build_model3, epochs=20, batch_size=64)

    cls = VotingClassifier(estimators=(['model1', model1],
                                       ['model2', model2],
                                       ['model3', model3]),
                           n_jobs=-1)
    cls = cls.fit(x_train, y_train)



    # for index, m in enumerate(models):
    #     weights_path = "./classification-voting-ensemble/sklearn_model" + str(index+1) +".h5"
    #     file = Path(weights_path)
    #     if not file.exists():
    #         history = m.fit(x_train, y_train,
    #                         epochs=20,
    #                         batch_size=64,
    #                         validation_split=0.3)
    #         draw_train_history(history)
    #         m.model.save_weights(weights_path)
    #         loss, accuracy = m.model.evaluate(x_test, y_test)
    #         print("model" + str(index + 1) + " test loss: {}, test accuracy: {}".format(loss, accuracy))
    #     else:
    #         m.model.load_weights(weights_path)
    #
    #     result = m.predict(x_test)
    #     print(result.shape)
    #     result = result.max(axis=1, keepdims=True)
    #     results.append(result)
    #
    # results = np.array(results)
    # print(results)
    # print(results.shape)