# 实现双弧形二分类

逻辑异或问题的成功解决，可以带给我们一定的信心，但是毕竟只有4个样本，还不能发挥出双层神经网络的真正能力。下面让我们一起来解决问题二，复杂的二分类问题。

## 代码实现

### 主过程代码

```python
if __name__ == '__main__':
    ......
    n_input = dataReader.num_feature
    n_hidden = 2
    n_output = 1
    eta, batch_size, max_epoch = 0.1, 5, 10000
    eps = 0.08

    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.BinaryClassifier, InitialMethod.Xavier)
    net = NeuralNet2(hp, "Arc_221")
    net.train(dataReader, 5, True)
    net.ShowTrainingTrace()
```

此处的代码有几个需要强调的细节：

* n\_input = dataReader.num\_feature，值为2，而且必须为2，因为只有两个特征值
* n\_hidden=2，这是人为设置的隐层神经元数量，可以是大于2的任何整数
* eps精度=0.08是后验知识，笔者通过测试得到的停止条件，用于方便案例讲解
* 网络类型是NetType.BinaryClassifier，指明是二分类网络

## 运行结果

经过快速的迭代，训练完毕后，会显示损失函数曲线和准确率曲线如图10-15。

![&#x56FE;10-15 &#x8BAD;&#x7EC3;&#x8FC7;&#x7A0B;&#x4E2D;&#x7684;&#x635F;&#x5931;&#x51FD;&#x6570;&#x503C;&#x548C;&#x51C6;&#x786E;&#x7387;&#x503C;&#x7684;&#x53D8;&#x5316;](../.gitbook/assets/image%20%28258%29.png)

蓝色的线条是小批量训练样本的曲线，波动相对较大，不必理会，因为批量小势必会造成波动。红色曲线是验证集的走势，可以看到二者的走势很理想，经过一小段时间的磨合后，从第200个epoch开始，两条曲线都突然找到了突破的方向，然后只用了50个epoch，就迅速达到指定精度。

同时在控制台会打印一些信息，最后几行如下：

```python
......
epoch=259, total_iteration=18719
loss_train=0.092687, accuracy_train=1.000000
loss_valid=0.074073, accuracy_valid=1.000000
W= [[ 8.88189429  6.09089509]
 [-7.45706681  5.07004428]]
B= [[ 1.99109895 -7.46281087]]
W= [[-9.98653838]
 [11.04185384]]
B= [[3.92199463]]
testing...
1.0
```

一共用了260个epoch，达到了指定的loss精度（0.08）时停止迭代。看测试集的情况，准确度1.0，即100%分类正确。

## 代码位置

原代码位置：[ch10, Level3](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch10-NonLinearBinaryClassification/Level3_DoubleArcClassifier.py)

个人代码：[**DoubleArcClassifier**](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/NonLinearBinaryClassification/DoubleArcClassifier.py)\*\*\*\*

## **keras实现**

```python
from HelperClass2.DataReader_2_0 import *

from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def load_data():
    train_data_name = "../data/ch10.train.npz"
    test_data_name = "../data/ch10.test.npz"

    dataReader = DataReader_2_0(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.Shuffle()
    dataReader.GenerateValidationSet()

    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    x_test, y_test = dataReader.XTest, dataReader.YTest
    x_val, y_val = dataReader.XDev, dataReader.YDev
    return x_train, y_train, x_test, y_test, x_val, y_val


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
    x_train, y_train, x_test, y_test, x_val, y_val = load_data()

    model = build_model()
    history = model.fit(x_train, y_train, epochs=500, batch_size=5, validation_data=(x_val, y_val))
    draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    weights = model.get_weights()
    print("weights: ", weights)
```

模型输出

```python
test loss: 0.3726512098312378, test accuracy: 0.8199999928474426
weights:  [array([[-0.06496473,  0.02829591],
       [-4.5003347 ,  4.043444  ]], dtype=float32), array([ 2.66075  , -2.1573546], dtype=float32), array([[-5.7543817],
       [ 4.9798098]], dtype=float32), array([0.25002602], dtype=float32)]
```

训练损失以及准确率曲线

![](../.gitbook/assets/image%20%28227%29.png)

