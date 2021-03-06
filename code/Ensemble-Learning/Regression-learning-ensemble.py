# -*- coding: utf-8 -*-#
'''
# Name:         Regression-ensemble
# Description:  
# Author:       super
# Date:         2020/6/5
'''

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import boston_housing
from keras.models import Model
from keras.layers import Input, Dense, concatenate
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
    inputs = Input(shape=(13, ))
    model1_1 = Dense(64, activation='relu')(inputs)
    model2_1 = Dense(128, activation='relu')(inputs)
    model3_1 = Dense(32, activation='relu')(inputs)
    model1_2 = Dense(32, activation='relu')(model1_1)
    model2_2 = Dense(64, activation='relu')(model2_1)
    model3_2 = Dense(16, activation='relu')(model3_1)
    model1_3 = Dense(1, activation='linear')(model1_2)
    model2_3 = Dense(1, activation='linear')(model2_2)
    model3_3 = Dense(1, activation='linear')(model3_2)
    con = concatenate([model1_3, model2_3, model3_3])
    output = Dense(1, activation='linear')(con)
    model = Model(inputs=inputs, outputs=output)
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
    model.save("regression-learning-ensemble.h5")

    loss = model.evaluate(x_test, y_test, batch_size=64)
    print("test loss: {}".format(loss))