# -*- coding: utf-8 -*-#
'''
# Name:         TuneParams2-keras
# Description:  
# Author:       super
# Date:         2020/5/24
'''

from HelperClass2.DataReader_2_0 import *

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam

from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt

def load_data():
    train_data_name = "../data/ch08.train.npz"
    test_data_name = "../data/ch08.test.npz"

    dataReader = DataReader_2_0(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.GenerateValidationSet()

    x_train, y_train, x_val, y_val = dataReader.XTrain, dataReader.YTrain, dataReader.XDev, dataReader.YDev
    x_test, y_test = dataReader.XTest, dataReader.YTest
    return x_train, y_train, x_val, y_val, x_test, y_test


def build_model(eta):
    model = Sequential()
    model.add(Dense(2, activation='sigmoid', input_shape=(1,)))
    model.add(Dense(1, activation='linear'))
    adam = Adam(lr=eta)
    model.compile(optimizer=adam,
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

    seed = 7
    np.random.seed(seed)

    eta = [0.3, 0.4]
    batch = [16, 32]
    epochs = [50]

    model = KerasRegressor(build_fn=build_model)

    param_grid = dict(batch_size=batch, eta=eta, epochs=epochs)
    grid = RandomizedSearchCV(estimator=model,
                              param_distributions=param_grid,
                              cv=5,
                              n_jobs=-1)
    grid_result = grid.fit(x_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("means:%f, std: %f with params: %r" % (mean, stdev, param))