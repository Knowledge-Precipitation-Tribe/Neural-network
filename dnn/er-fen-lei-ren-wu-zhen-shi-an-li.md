# 二分类任务真实案例

我们用一个真实的数据级来实现一个二分类任务：收入调查与预测，即给定一个居民的各种情况，如工作、家庭、学历等，来预测该居民的年收入是否可以大于$$50K/年$$，所以大于$$50K$$的就是正例，而小于等于$$50K$$的就是负例。

## 准备数据

此数据集是从1994 Census数据库中提取的$$^{[1]}$$。

### 数据字段解读

标签值：&gt;50K，&lt;=50K。

属性字段：

* age，年龄：连续值
* workclass，工作性质：枚举型，类似私企、政府之类的
* fnlwgt，权重：连续值
* education，教育程度：枚举型，如学士、硕士等
* education-num，受教育的时长：连续值
* marital-status，婚姻状况：枚举型，已婚、未婚、离异等
* occupation，职业：枚举型，包含的种类很多，如技术支持、维修工、销售、农民渔民、军人等
* relationship，家庭角色：枚举型，丈夫、妻子等
* sex，性别：枚举型
* capital-gain，资本收益：连续值
* capitial-loss，资本损失：连续值
* hours-per-week，每周工作时长：连续值
* native-country，祖籍：枚举型

### 数据处理

数据分析和数据处理实际上是一门独立的课，超出类本书的范围，所以我们只做一些简单的数据处理，以便神经网络可以用之训练。

对于连续值，我们可以直接使用原始数据。对于枚举型，我们需要把它们转成连续值。以性别举例，Female=0，Male=1即可。对于其它枚举型，都可以用从0开始的整数编码。

一个小技巧是利用python的list功能，取元素下标，即可以作为整数编码：

```python
sex_list = ["Female", "Male"]
array_x[0,9] = sex_list.index(row[9].strip())
```

strip\(\)是trim掉前面的空格，因为是csv格式，读出来会是这个样子："\_Female"，前面总有个空格。index是取列表下标，这样对于字符串"Female"取出的下标为0，对于字符串"Male"取出的下标为1。

把所有数据按行保存到numpy数组中，最后用npz格式存储：

```python
np.savez(data_npz, data=self.XData, label=self.YData)
```

原始数据已经把train data和test data分开了，所以我们针对两个数据集分别调用数据处理过程一次，保存为Income\_Train.npz和Income\_Test.npz。

### 加载数据

```python
train_file = "../../Data/ch14.Income.train.npz"
test_file = "../../Data/ch14.Income.test.npz"

def LoadData():
    dr = DataReader_2_0(train_file, test_file)
    dr.ReadData()
    dr.NormalizeX()
    dr.Shuffle()
    dr.GenerateValidationSet()
    return dr
```

因为属性字段众多，取值范围相差很大，所以一定要先调用NormalizeX\(\)函数做归一化。由于是二分类问题，在做数据处理时，我们已经把大于$$50K$$标记为1，小于等于$$50K$$标记为0，所以不需要做标签值的归一化。

## 搭建模型

我们搭建一个网络结构，不同的是为了完成二分类任务，在最后接一个Logistic函数。

![&#x56FE;14-10 &#x5B8C;&#x6210;&#x4E8C;&#x5206;&#x7C7B;&#x771F;&#x5B9E;&#x6848;&#x4F8B;&#x7684;&#x62BD;&#x8C61;&#x6A21;&#x578B;](../.gitbook/assets/image%20%28312%29.png)

```python
def model(dr):
    num_input = dr.num_feature
    num_hidden1 = 32
    num_hidden2 = 16
    num_hidden3 = 8
    num_hidden4 = 4
    num_output = 1

    max_epoch = 100
    batch_size = 16
    learning_rate = 0.1

    params = HyperParameters_4_0(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.BinaryClassifier,
        init_method=InitialMethod.MSRA,
        stopper=Stopper(StopCondition.StopDiff, 1e-3))

    net = NeuralNet_4_0(params, "Income")

    fc1 = FcLayer_1_0(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    a1 = ActivationLayer(Relu())
    net.add_layer(a1, "relu1")
    ......
    fc5 = FcLayer_1_0(num_hidden4, num_output, params)
    net.add_layer(fc5, "fc5")
    logistic = ClassificationLayer(Logistic())
    net.add_layer(logistic, "logistic")

    net.train(dr, checkpoint=1, need_test=True)
    return net
```

超参数说明：

1. 学习率=0.1
2. 最大epoch=100
3. 批大小=16
4. 二分类网络类型
5. MSRA初始化
6. 相对误差停止条件1e-3

net.train\(\)函数是一个阻塞函数，只有当训练完毕后才返回。

## 训练结果

下图左边是损失函数图，右边是准确率图。忽略测试数据的波动，只看红色的验证集的趋势，损失函数值不断下降，准确率不断上升。

为什么不把max\_epoch设置为更大的数字，比如1000，以便得到更好的结果呢？实际上，训练更多的次数，因为过拟合的风险，不会得到更好的结果。有兴趣的读者可以自己试验一下。

![&#x56FE;14-11 &#x8BAD;&#x7EC3;&#x8FC7;&#x7A0B;&#x4E2D;&#x635F;&#x5931;&#x51FD;&#x6570;&#x503C;&#x548C;&#x51C6;&#x786E;&#x7387;&#x7684;&#x53D8;&#x5316;](../.gitbook/assets/image%20%28313%29.png)

下面是最后的打印输出：

```python
......
epoch=99, total_iteration=169699
loss_train=0.296219, accuracy_train=0.800000
loss_valid=nan, accuracy_valid=0.838859
time used: 29.866002321243286
testing...
0.8431606905710491
```

最后用独立的测试集得到的结果是84%，与该数据集相关的其它论文相比，已经是一个不错的成绩了。

## 代码位置

原代码位置：[ch14, Level4](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch14-DnnBasic/Level4_IncomeClassifier.py)

个人代码：[**IncomeClassifier**](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/DNN/IncomeClassifier.py)\*\*\*\*

## keras实现

```python
from MiniFramework.DataReader_2_0 import *

from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_data():
    train_data_name = "../data/ch14.Income.train.npz"
    test_data_name = "../data/ch14.Income.test.npz"

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
    model.add(Dense(32, activation='relu', input_shape=(14, )))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
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
    history = model.fit(x_train, y_train, epochs=20, batch_size=16, validation_data=(x_val, y_val))
    draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    weights = model.get_weights()
    print("weights: ", weights)
```

模型输出

```python
test loss: 0.3331818307855056, test accuracy: 0.8428950905799866
```

模型损失和准确率曲线

![](../.gitbook/assets/image%20%28328%29.png)

## 参考资料

\[1\] Dua, D. and Graff, C. \(2019\). UCI Machine Learning Repository \[[http://archive.ics.uci.edu/ml](http://archive.ics.uci.edu/ml)\]. Irvine, CA: University of California, School of Information and Computer Science.

Ronny Kohavi and Barry Becker Data Mining and Visualization Silicon Graphics. e-mail: ronnyk '@' sgi.com for questions.

[https://archive.ics.uci.edu/ml/datasets/Census+Income](https://archive.ics.uci.edu/ml/datasets/Census+Income)

