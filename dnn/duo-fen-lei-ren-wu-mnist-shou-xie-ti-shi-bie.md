# 多分类任务 - MNIST手写体识别

## 数据读取

MNIST数据本身是图像格式的，我们用mode="vector"去读取，转变成矢量格式的。

```python
def LoadData():
    print("reading data...")
    dr = MnistImageDataReader(mode="vector")
    ......
```

## 搭建模型

一共4个隐层，都用ReLU激活函数连接，最后的输出层接Softmax分类函数。

![&#x56FE;14-18 &#x5B8C;&#x6210;MNIST&#x5206;&#x7C7B;&#x4EFB;&#x52A1;&#x7684;&#x62BD;&#x8C61;&#x6A21;&#x578B;](../.gitbook/assets/image%20%28315%29.png)

以下是主要的参数设置：

```python
if __name__ == '__main__':
    dataReader = LoadData()
    num_feature = dataReader.num_feature
    num_example = dataReader.num_example
    num_input = num_feature
    num_hidden1 = 128
    num_hidden2 = 64
    num_hidden3 = 32
    num_hidden4 = 16
    num_output = 10
    max_epoch = 10
    batch_size = 64
    learning_rate = 0.1

    params = HyperParameters_4_0(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        stopper=Stopper(StopCondition.StopLoss, 0.12))

    net = NeuralNet_4_0(params, "MNIST")

    fc1 = FcLayer_1_0(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "r1")
    ......
    fc5 = FcLayer_1_0(num_hidden4, num_output, params)
    net.add_layer(fc5, "fc5")
    softmax = ClassificationLayer(Softmax())
    net.add_layer(softmax, "softmax")

    net.train(dataReader, checkpoint=0.05, need_test=True)
    net.ShowLossHistory(xcoord=XCoordinate.Iteration)
```

## 运行结果

我们设计的停止条件是绝对Loss值达到0.12时，所以迭代到6个epoch时，达到了0.119的损失值，就停止训练了。

图14-19 训练过程中损失函数值和准确率的变化

图14-19是训练过程图示，下面是最后几行的打印输出。

```python
......
epoch=6, total_iteration=5763
loss_train=0.005559, accuracy_train=1.000000
loss_valid=0.119701, accuracy_valid=0.971667
time used: 17.500738859176636
save parameters
testing...
0.9697
```

最后用测试集得到的准确率为96.97%。

## 代码位置

原代码位置：[ch14, Level6](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch14-DnnBasic/Level6_MnistClassifier.py)

个人代码：[**MnistClassifier**](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/DNN/MnistClassifier.py)\*\*\*\*

## keras实现

```python
from ExtendedDataReader.MnistImageDataReader import *

from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_data():
    dataReader = MnistImageDataReader(mode="vector")
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.NormalizeY(NetType.MultipleClassifier)
    dataReader.GenerateValidationSet(k=20)

    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    x_test, y_test = dataReader.XTest, dataReader.YTest
    x_val, y_val = dataReader.XDev, dataReader.YDev

    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_val = x_val.reshape(x_val.shape[0], 28 * 28)


    return x_train, y_train, x_test, y_test, x_val, y_val

def build_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(784, )))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
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


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, x_val, y_val = load_data()
    # print(x_train.shape)
    # print(x_test.shape)
    # print(x_val.shape)

    model = build_model()
    history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_val, y_val))
    draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    weights = model.get_weights()
    print("weights: ", weights)
```

模型输出

```python
test loss: 0.11646892445675121, test accuracy: 0.9768999814987183
```

模型损失以及准确率曲线

![](../.gitbook/assets/image%20%28318%29.png)

