# 曲线拟合

在上一节我们已经写好了神经网络的核心模块及其辅助功能，现在我们先来做一下正弦曲线的拟合，然后再试验复合函数的曲线拟合。

## 正弦曲线的拟合

### 隐层只有一个神经元的情况

令n\_hidden=1，并指定模型名称为"sin\_111"，训练过程见图9-10。图9-11为拟合效果图。

![&#x56FE;9-10 &#x8BAD;&#x7EC3;&#x8FC7;&#x7A0B;&#x4E2D;&#x635F;&#x5931;&#x51FD;&#x6570;&#x503C;&#x548C;&#x51C6;&#x786E;&#x7387;&#x7684;&#x53D8;&#x5316;](../.gitbook/assets/image%20%28174%29.png)

![&#x56FE;9-11 &#x4E00;&#x4E2A;&#x795E;&#x7ECF;&#x5143;&#x7684;&#x62DF;&#x5408;&#x6548;&#x679C;](../.gitbook/assets/image%20%28158%29.png)

从图9-10可以看到，损失值到0.04附近就很难下降了。图9-11中，可以看到只有中间线性部分拟合了，两端的曲线部分没有拟合。

```python
......
epoch=4999, total_iteration=224999
loss_train=0.015787, accuracy_train=0.943360
loss_valid=0.038609, accuracy_valid=0.821760
testing...
0.8575700023301912
```

打印输出最后的测试集精度值为85.7%，不是很理想。所以隐层1个神经元是基本不能工作的，这只比单层神经网络的线性拟合强一些，距离目标还差很远。

### 隐层有两个神经元的情况

```python
if __name__ == '__main__':
    ......
    n_input, n_hidden, n_output = 1, 2, 1
    eta, batch_size, max_epoch = 0.05, 10, 5000
    eps = 0.001
    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.Fitting, InitialMethod.Xavier)
    net = NeuralNet2(hp, "sin_121")
    #net.LoadResult()
    net.train(dataReader, 50, True)
    ......
```

初始化神经网络类的参数有两个，第一个是超参组合hp，第二个是指定模型专有名称，以便把结果保存在名称对应的子目录中。保存训练结果的代码在训练结束后自动调用，但是如果想加载历史训练结果，需要在主过程中手动调用，比如上面代码中注释的那一行：net.LoadResult\(\)。这样的话，如果下次再训练，就可以在以前的基础上继续训练，不必从头开始。

注意在主过程代码中，我们指定了n\_hidden=2，意为隐层神经元数量为2。

### 运行结果

图9-12为损失函数曲线和验证集精度曲线，都比较正常。而2个神经元的网络损失值可以达到0.004，少一个数量级。验证集精度到82%左右，而2个神经元的网络可以达到97%。图9-13为拟合效果图。

![&#x56FE;9-12 &#x4E24;&#x4E2A;&#x795E;&#x7ECF;&#x5143;&#x7684;&#x8BAD;&#x7EC3;&#x8FC7;&#x7A0B;&#x4E2D;&#x635F;&#x5931;&#x51FD;&#x6570;&#x503C;&#x548C;&#x51C6;&#x786E;&#x7387;&#x7684;&#x53D8;&#x5316;](../.gitbook/assets/image%20%28176%29.png)

![&#x56FE;9-13 &#x4E24;&#x4E2A;&#x795E;&#x7ECF;&#x5143;&#x7684;&#x62DF;&#x5408;&#x6548;&#x679C;](../.gitbook/assets/image%20%28141%29.png)

再看下面的打印输出结果，最后测试集的精度为98.8%。如果需要精度更高的话，可以增加迭代次数。

```python
......
epoch=4999, total_iteration=224999
loss_train=0.007681, accuracy_train=0.971567
loss_valid=0.004366, accuracy_valid=0.979845
testing...
0.9881468747638157
```

## 复合函数的拟合

基本过程与正弦曲线相似，区别是这个例子要复杂不少，所以首先需要耐心，增大max\_epoch的数值，多迭代几次。其次需要精心调参，找到最佳参数组合。

### 隐层只有两个神经元的情况

![&#x56FE;9-14 &#x4E24;&#x4E2A;&#x795E;&#x7ECF;&#x5143;&#x7684;&#x62DF;&#x5408;&#x6548;&#x679C;](../.gitbook/assets/image%20%28165%29.png)

图9-14是两个神经元的拟合效果图，拟合情况很不理想，和正弦曲线只用一个神经元的情况类似。观察打印输出的损失值，有波动，久久徘徊在0.003附近不能下降，说明网络能力不够。

```python
epoch=99999, total_iteration=8999999
loss_train=0.000751, accuracy_train=0.968484
loss_valid=0.003200, accuracy_valid=0.795622
testing...
0.8641114405898856
```

### 隐层有三个神经元的情况

```python
if __name__ == '__main__':
    ......
    n_input, n_hidden, n_output = 1, 3, 1
    eta, batch_size, max_epoch = 0.5, 10, 10000
    eps = 0.001
    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.Fitting, InitialMethod.Xavier)
    net = NeuralNet2(hp, "model_131")
    ......
```

### 运行结果

图9-15为损失函数曲线和验证集精度曲线，都比较正常。图9-16是拟合效果。

![&#x56FE;9-15 &#x4E09;&#x4E2A;&#x795E;&#x7ECF;&#x5143;&#x7684;&#x8BAD;&#x7EC3;&#x8FC7;&#x7A0B;&#x4E2D;&#x635F;&#x5931;&#x51FD;&#x6570;&#x503C;&#x548C;&#x51C6;&#x786E;&#x7387;&#x7684;&#x53D8;&#x5316;](../.gitbook/assets/image%20%28146%29.png)

![&#x56FE;9-16 &#x4E09;&#x4E2A;&#x795E;&#x7ECF;&#x5143;&#x7684;&#x62DF;&#x5408;&#x6548;&#x679C;](../.gitbook/assets/image%20%28147%29.png)

再看下面的打印输出结果，最后测试集的精度为97.6%，已经令人比较满意了。如果需要精度更高的话，可以增加迭代次数。

```python
......
epoch=4199, total_iteration=377999
loss_train=0.001152, accuracy_train=0.963756
loss_valid=0.000863, accuracy_valid=0.944908
testing...
0.9765910104463337
```

以下就是笔者找到的最佳组合：

* 隐层3个神经元
* 学习率=0.5
* 批量=10

## 广义的回归/拟合

至此我们用两个可视化的例子完成了曲线拟合，验证了万能近似定理。但是，神经网络不是设计专门用于曲线拟合的，这只是牛刀小试而已，我们用简单的例子讲解了神经网络的功能，但是此功能完全可以用于多变量的复杂非线性回归。

**“曲线”在这里是一个广义的概念，它不仅可以代表二维平面上的数学曲线，也可以代表工程实践中的任何拟合问题，比如房价预测问题，影响房价的自变量可以达到20个左右，显然已经超出了线性回归的范畴，此时我们可以用多层神经网络来做预测。**在后面我们会讲解这样的例子。

简言之，只要是数值拟合问题，确定不能用线性回归的话，都可以用非线性回归来尝试解决。

## 代码位置

原代码位置：[ch09, Level3](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch09-NonLinearRegression/Level3_NN_Sin.py), [Level4](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch09-NonLinearRegression/Level4_NN_Complex.py)

个人代码：[NN\_Sin](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/NonLinearRegression/NN_Sin.py), [NN\_Complex](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/NonLinearRegression/NN_Complex.py)

## keras实现

### 拟合Sin曲线

```python
from HelperClass2.DataReader_2_0 import *

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
from contextlib import redirect_stdout

import os

def load_data():
    train_data_name = "../data/ch08.train.npz"
    test_data_name = "../data/ch08.test.npz"

    dataReader = DataReader_2_0(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.GenerateValidationSet()

    x_train, y_train, x_val, y_val = dataReader.XTrain, dataReader.YTrain, dataReader.XDev, dataReader.YDev
    x_test, y_test = dataReader.XTest, dataReader.YTest
    return x_train, y_train, x_val, y_val, x_test, y_test


def build_model():
    model = Sequential()
    model.add(Dense(2, activation='sigmoid', input_shape=(1,)))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='Adam',
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

    model_path = "nn_sin.h5"
    model_weights_path = "nn_sin_weights.h5"

    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = build_model()
        early_stopping = EarlyStopping(monitor='loss', patience=100)
        history = model.fit(x_train, y_train,
                            epochs=5000,
                            batch_size=32,
                            callbacks=[early_stopping],
                            validation_data=(x_val, y_val))
        draw_train_history(history)
        loss= model.evaluate(x_test, y_test, batch_size=32)
        print("test loss: {}".format(loss))
        model.save(model_path)
        model.save_weights(model_weights_path)

    model_summary_path = "nn_sin_summary.txt"
    with open(model_summary_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
```

模型训练过程中的损失

![](../.gitbook/assets/image%20%28187%29.png)

### 拟合复合函数曲线

```python
from HelperClass2.DataReader_2_0 import *

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
from contextlib import redirect_stdout

import os

def load_data():
    train_data_name = "../data/ch09.train.npz"
    test_data_name = "../data/ch09.test.npz"

    dataReader = DataReader_2_0(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.GenerateValidationSet()

    x_train, y_train, x_val, y_val = dataReader.XTrain, dataReader.YTrain, dataReader.XDev, dataReader.YDev
    x_test, y_test = dataReader.XTest, dataReader.YTest
    return x_train, y_train, x_val, y_val, x_test, y_test


def build_model():
    model = Sequential()
    model.add(Dense(3, activation='sigmoid', input_shape=(1,)))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='Adam',
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

    model_path = "nn_complex.h5"
    model_weights_path = "nn_complex_weights.h5"

    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = build_model()
        early_stopping = EarlyStopping(monitor='loss', patience=100)
        history = model.fit(x_train, y_train,
                            epochs=5000,
                            batch_size=32,
                            callbacks=[early_stopping],
                            validation_data=(x_val, y_val))
        draw_train_history(history)
        loss= model.evaluate(x_test, y_test, batch_size=32)
        print("test loss: {}".format(loss))
        model.save(model_path)
        model.save_weights(model_weights_path)

    model_summary_path = "nn_complex_summary.txt"
    with open(model_summary_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
```

模型训练过程中的损失

![](../.gitbook/assets/image%20%28144%29.png)

