# -*- coding: utf-8 -*-#
'''
# Name:         MultipleClassification-keras
# Description:  
# Author:       super
# Date:         2020/5/22
'''

import numpy as np

from HelperClass.NeuralNet_1_2 import *
from HelperClass.DataReader_1_3 import *

from keras.models import Sequential
from keras.layers import Dense

def load_data(num_category, path):
    reader = DataReader_1_3(path)
    reader.ReadData()
    reader.NormalizeX()
    reader.ToOneHot(num_category, base=1)

    xt_raw = np.array([5, 1, 7, 6, 5, 6, 2, 7]).reshape(4, 2)
    x_test = reader.NormalizePredicateData(xt_raw)

    return reader.XTrain, reader.YTrain, x_test


def build_model():
    model = Sequential()
    model.add(Dense(3, activation='softmax', input_shape=(2,)))
    model.compile(optimizer='SGD', loss='categorical_crossentropy')
    return model

if __name__ == '__main__':
    path = "../data/ch07.npz"
    x_train, y_train, x_test = load_data(num_category=3, path=path)
    # print(x_train)
    # print(y_train)

    model = build_model()
    model.fit(x_train, y_train, epochs=100, batch_size=10)
    w, b = model.layers[0].get_weights()
    print(w)
    print(b)

    output = model.predict(x_test)
    r = np.argmax(output, axis=1) + 1
    print("output=", output)
    print("r=", r)