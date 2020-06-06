# -*- coding: utf-8 -*-#
'''
# Name:         Classification-learning-ensemble
# Description:  
# Author:       super
# Date:         2020/6/5
'''

import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, concatenate

from sklearn.preprocessing import OneHotEncoder

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


def build_model():
    inputs = Input(shape=(784, ))
    model1_1 = Dense(64, activation='relu')(inputs)
    model2_1 = Dense(128, activation='relu')(inputs)
    model3_1 = Dense(32, activation='relu')(inputs)
    model1_2 = Dense(32, activation='relu')(model1_1)
    model2_2 = Dense(64, activation='relu')(model2_1)
    model3_2 = Dense(16, activation='relu')(model3_1)
    model1_3 = Dense(16, activation='relu')(model1_2)
    model2_3 = Dense(32, activation='relu')(model2_2)
    model3_3 = Dense(8, activation='relu')(model3_2)
    con = concatenate([model1_3, model2_3, model3_3])
    output = Dense(10, activation='softmax')(con)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()
    history = model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=64,
                        validation_split=0.3)
    draw_train_history(history)
    model.save("classification-learning-ensemble.h5")

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))