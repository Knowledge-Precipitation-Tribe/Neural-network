# 二分类任务功能测试

## 搭建模型

同样是一个双层神经网络，但是最后一层要接一个Logistic二分类函数来完成二分类任务，如图14-7所示。

![&#x56FE;14-7 &#x5B8C;&#x6210;&#x975E;&#x7EBF;&#x6027;&#x4E8C;&#x5206;&#x7C7B;&#x6559;&#x5B66;&#x6848;&#x4F8B;&#x7684;&#x62BD;&#x8C61;&#x6A21;&#x578B;](../.gitbook/assets/image%20%28310%29.png)

```python
def model(dataReader):
    num_input = 2
    num_hidden = 3
    num_output = 1

    max_epoch = 1000
    batch_size = 5
    learning_rate = 0.1

    params = HyperParameters_4_0(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.BinaryClassifier,
        init_method=InitialMethod.Xavier,
        stopper=Stopper(StopCondition.StopLoss, 0.02))

    net = NeuralNet_4_0(params, "Arc")

    fc1 = FcLayer_1_0(num_input, num_hidden, params)
    net.add_layer(fc1, "fc1")
    sigmoid1 = ActivationLayer(Sigmoid())
    net.add_layer(sigmoid1, "sigmoid1")
    
    fc2 = FcLayer_1_0(num_hidden, num_output, params)
    net.add_layer(fc2, "fc2")
    logistic = ClassificationLayer(Logistic())
    net.add_layer(logistic, "logistic")

    net.train(dataReader, checkpoint=10, need_test=True)
    return net
```

超参数说明：

1. 输入层神经元数为2
2. 隐层的神经元数为3，使用Sigmoid激活函数
3. 由于是二分类任务，所以输出层只有一个神经元，用Logistic做二分类函数
4. 最多训练1000轮
5. 批大小=5
6. 学习率=0.1
7. 绝对误差停止条件=0.02

## 运行结果

![&#x56FE;14-8 &#x8BAD;&#x7EC3;&#x8FC7;&#x7A0B;&#x4E2D;&#x635F;&#x5931;&#x51FD;&#x6570;&#x503C;&#x548C;&#x51C6;&#x786E;&#x7387;&#x7684;&#x53D8;&#x5316;](../.gitbook/assets/image%20%28320%29.png)

图14-8是训练记录，再看下面的打印输出结果：

```python
......
epoch=419, total_iteration=30239
loss_train=0.010094, accuracy_train=1.000000
loss_valid=0.019141, accuracy_valid=1.000000
time used: 2.149379253387451
testing...
1.0
```

最后的testing...的结果是1.0，表示100%正确，这初步说明mini框架在这个基本case上工作得很好。图14-9所示的分类效果也不错。

![&#x56FE;14-9 &#x5206;&#x7C7B;&#x6548;&#x679C;](../.gitbook/assets/image%20%28309%29.png)

## 代码位置

原代码位置：[ch14, Level3](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch14-DnnBasic/Level3_ch10.py)

个人代码：[**dnn\_classification**](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/DNN/dnn_classification.py)\*\*\*\*

## keras实现

```python
from MiniFramework.DataReader_2_0 import *

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
    model.add(Dense(3, activation='sigmoid', input_shape=(2, )))
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
    history = model.fit(x_train, y_train, epochs=200, batch_size=5, validation_data=(x_val, y_val))
    draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    weights = model.get_weights()
    print("weights: ", weights)
```

模型输出

```python
test loss: 0.3908280086517334, test accuracy: 0.8100000023841858
weights:  [array([[-0.40774214, -0.3335594 ,  0.46907774],
       [-2.6843045 ,  3.6533718 , -4.166602  ]], dtype=float32), array([ 1.0028745, -1.3372192,  1.7076769], dtype=float32), array([[-2.6436245],
       [ 3.5234995],
       [-4.228298 ]], dtype=float32), array([0.3786795], dtype=float32)]
```

模型损失以及准确率曲线

![](../.gitbook/assets/image%20%28319%29.png)

