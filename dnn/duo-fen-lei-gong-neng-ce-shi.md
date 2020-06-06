# 多分类功能测试

## 搭建模型一

### 模型

使用Sigmoid做为激活函数的两层网络，如图14-12。

![&#x56FE;14-12 &#x5B8C;&#x6210;&#x975E;&#x7EBF;&#x6027;&#x591A;&#x5206;&#x7C7B;&#x6559;&#x5B66;&#x6848;&#x4F8B;&#x7684;&#x62BD;&#x8C61;&#x6A21;&#x578B;](../.gitbook/assets/image%20%28329%29.png)

### 代码

```python
def model_sigmoid(num_input, num_hidden, num_output, hp):
    net = NeuralNet_4_0(hp, "chinabank_sigmoid")

    fc1 = FcLayer_1_0(num_input, num_hidden, hp)
    net.add_layer(fc1, "fc1")
    s1 = ActivationLayer(Sigmoid())
    net.add_layer(s1, "Sigmoid1")

    fc2 = FcLayer_1_0(num_hidden, num_output, hp)
    net.add_layer(fc2, "fc2")
    softmax1 = ClassificationLayer(Softmax())
    net.add_layer(softmax1, "softmax1")

    net.train(dataReader, checkpoint=50, need_test=True)
    net.ShowLossHistory()
    
    ShowResult(net, hp.toString())
    ShowData(dataReader)
```

### 超参数说明

1. 隐层8个神经元
2. 最大epoch=5000
3. 批大小=10
4. 学习率0.1
5. 绝对误差停止条件=0.08
6. 多分类网络类型
7. 初始化方法为Xavier

net.train\(\)函数是一个阻塞函数，只有当训练完毕后才返回。

### 运行结果

训练过程如图14-13所示，分类效果如图14-14所示。

![&#x56FE;14-13 &#x8BAD;&#x7EC3;&#x8FC7;&#x7A0B;&#x4E2D;&#x635F;&#x5931;&#x51FD;&#x6570;&#x503C;&#x548C;&#x51C6;&#x786E;&#x7387;&#x7684;&#x53D8;&#x5316;](../.gitbook/assets/image%20%28316%29.png)

![&#x56FE;14-14 &#x5206;&#x7C7B;&#x6548;&#x679C;&#x56FE;](../.gitbook/assets/image%20%28325%29.png)

## 搭建模型二

### 模型

使用ReLU做为激活函数的三层网络，如图14-15。

![&#x56FE;14-15 &#x4F7F;&#x7528;ReLU&#x51FD;&#x6570;&#x62BD;&#x8C61;&#x6A21;&#x578B;](../.gitbook/assets/image%20%28330%29.png)

用两层网络也可以实现，但是使用ReLE函数时，训练效果不是很稳定，用三层比较保险。

### 代码

```python
def model_relu(num_input, num_hidden, num_output, hp):
    net = NeuralNet_4_0(hp, "chinabank_relu")

    fc1 = FcLayer_1_0(num_input, num_hidden, hp)
    net.add_layer(fc1, "fc1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "Relu1")

    fc2 = FcLayer_1_0(num_hidden, num_hidden, hp)
    net.add_layer(fc2, "fc2")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "Relu2")

    fc3 = FcLayer_1_0(num_hidden, num_output, hp)
    net.add_layer(fc3, "fc3")
    softmax = ClassificationLayer(Softmax())
    net.add_layer(softmax, "softmax")

    net.train(dataReader, checkpoint=50, need_test=True)
    net.ShowLossHistory()
    
    ShowResult(net, hp.toString())
    ShowData(dataReader)    
```

### 超参数说明

1. 隐层8个神经元
2. 最大epoch=5000
3. 批大小=10
4. 学习率0.1
5. 绝对误差停止条件=0.08
6. 多分类网络类型
7. 初始化方法为MSRA

### 运行结果

训练过程如图14-16所示，分类效果如图14-17所示。

![&#x56FE;14-16 &#x8BAD;&#x7EC3;&#x8FC7;&#x7A0B;&#x4E2D;&#x635F;&#x5931;&#x51FD;&#x6570;&#x503C;&#x548C;&#x51C6;&#x786E;&#x7387;&#x7684;&#x53D8;&#x5316;](../.gitbook/assets/image%20%28314%29.png)

![&#x56FE;14-17 &#x5206;&#x7C7B;&#x6548;&#x679C;&#x56FE;](../.gitbook/assets/image%20%28327%29.png)

## 比较

表14-1比较一下使用不同的激活函数的分类效果图。

表14-1 使用不同的激活函数的分类结果比较

| Sigmoid | ReLU |
| :--- | :--- |
| ![](../.gitbook/assets/image%20%28325%29.png)  | ![](../.gitbook/assets/image%20%28327%29.png)  |

可以看到左图中的边界要平滑许多，这也就是ReLU和Sigmoid的区别，ReLU是用分段线性拟合曲线，Sigmoid有真正的曲线拟合能力。但是Sigmoid也有缺点，看分类的边界，使用ReLU函数的分类边界比较清晰，而使用Sigmoid函数的分类边界要平缓一些，过渡区较宽。

用一句简单的话来描述二者的差别：Relu能直则直，对方形边界适用；Sigmoid能弯则弯，对圆形边界适用。

## 代码位置

原代码位置：[ch14, Level5](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch14-DnnBasic/Level5_ch11.py)

个人代码：[**dnn\_multiClassification**](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/DNN/dnn_multiClassification.py)\*\*\*\*

## **keras实现**

```python
from MiniFramework.DataReader_2_0 import *

from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_data():
    train_file = "../data/ch11.train.npz"
    test_file = "../data/ch11.test.npz"

    dataReader = DataReader_2_0(train_file, test_file)
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.NormalizeY(NetType.MultipleClassifier, base=1)
    dataReader.Shuffle()
    dataReader.GenerateValidationSet()

    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    x_test, y_test = dataReader.XTest, dataReader.YTest
    x_val, y_val = dataReader.XDev, dataReader.YDev

    return x_train, y_train, x_test, y_test, x_val, y_val

def build_model():
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(2, )))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))
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

    model = build_model()
    history = model.fit(x_train, y_train, epochs=100, batch_size=10, validation_data=(x_val, y_val))
    draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    weights = model.get_weights()
    print("weights: ", weights)
```

 **模型输出**

```python
test loss: 0.17712581551074982, test accuracy: 0.9480000138282776
weights:  [array([[ 0.5933495 ,  1.4001606 , -0.46128672, -1.0328066 ,  0.4704834 ,
        -0.13881624, -1.5763694 ,  0.39080548],
       [-1.368007  ,  0.6859568 , -0.73229355,  0.7306851 ,  0.7245091 ,
         0.07994052, -0.24856903, -1.7596178 ]], dtype=float32), array([ 0.45760065, -0.15192626,  0.        ,  0.07134774, -0.5471344 ,
        0.8713162 ,  0.70304626,  0.35331267], dtype=float32), array([[-0.44358087,  1.4647273 ,  0.6838032 ,  1.3377259 ,  0.9623304 ,
         0.801778  ,  0.07605773,  0.3418852 ],
       [-0.4460683 ,  0.34761527,  1.0513452 , -0.06174321, -1.0383745 ,
         0.6795882 ,  0.10854041,  0.7523184 ],
       [-0.10631716,  0.11873323, -0.6039761 ,  0.25613695, -0.52250814,
        -0.30054256, -0.21584505, -0.1406537 ],
       [-0.16741991,  1.4979596 ,  2.0585673 ,  1.7580718 , -0.12573877,
         1.0612497 ,  0.3230644 , -0.5291618 ],
       [-0.5735318 ,  1.6553403 ,  1.7515256 ,  2.3439772 , -0.6199714 ,
         1.8710839 , -0.479978  ,  0.32344452],
       [ 0.05696827, -0.42794174, -0.84942245,  0.140646  ,  0.10621271,
        -0.6504364 , -0.46572435,  1.5581474 ],
       [-0.5257491 ,  3.3067589 ,  0.88320696,  3.685748  ,  1.4463454 ,
         1.6930596 , -0.4242273 ,  0.01312767],
       [-0.17542031,  2.562144  ,  0.5151277 ,  2.2590203 ,  0.89615613,
         0.673661  , -0.22245789, -0.5107478 ]], dtype=float32), array([ 0.        , -0.279784  , -0.7419632 , -0.40435618, -0.25443217,
       -0.5012585 , -0.05188801,  1.19713   ], dtype=float32), array([[-0.70594245, -0.10075217, -0.66060597],
       [-4.915914  ,  0.15276138,  2.171216  ],
       [-4.7157874 , -1.9014189 ,  2.9048522 ],
       [-4.163957  ,  0.42147762,  1.3650419 ],
       [-2.2925568 , -2.7893898 ,  3.548446  ],
       [-4.951117  , -1.8089052 ,  2.7878227 ],
       [-0.04061006, -0.04594503, -0.29654002],
       [ 1.1871618 ,  0.6122024 , -1.6027299 ]], dtype=float32), array([ 1.4912351 ,  0.29161552, -1.5535753 ], dtype=float32)]

Process finished with exit code 0

```

**模型损失以及准确率曲线**

![](../.gitbook/assets/image%20%28324%29.png)

