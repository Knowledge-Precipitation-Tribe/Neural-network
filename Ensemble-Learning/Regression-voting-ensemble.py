# -*- coding: utf-8 -*-#
'''
# Name:         Regression-voting-ensemble
# Description:  
# Author:       super
# Date:         2020/6/6
'''
import numpy as np

from keras.datasets import boston_housing
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor


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


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(13, )))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam',
                  loss='mean_squared_error')
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()

    model1 = KerasRegressor(build_fn=build_model, epochs=100, batch_size=64)
    model1._estimator_type = "regressor"
    model2 = KerasRegressor(build_fn=build_model, epochs=100, batch_size=64)
    model2._estimator_type = "regressor"
    model3 = KerasRegressor(build_fn=build_model, epochs=100, batch_size=64)
    model3._estimator_type = "regressor"

    cls = VotingRegressor(estimators=(['model1', model1],
                                       ['model2', model2],
                                       ['model3', model3]),
                          n_jobs=-1)
    cls.fit(x_train, y_train)

    # print(cls.score(x_test, y_test))