# 回归任务 - 房价预测

## 数据

数据集来自：[https://www.kaggle.com/harlfoxem/housesalesprediction](https://www.kaggle.com/harlfoxem/housesalesprediction)

此数据集是King County地区2014年五月至2015年五月的房屋销售信息，适合于训练回归模型。

### 数据字段解读

* id：唯一id
* date：售出日期
* price：售出价格（标签值）
* bedrooms：卧室数量
* bathrooms：浴室数量
* sqft\_living：居住面积
* sqft\_lot：停车场面积
* floors：楼层数
* waterfront：泳池
* view：有多少次看房记录
* condition：房屋状况
* grade：评级
* sqft\_above：地面上的面积
* sqft\_basement：地下室的面积
* yr\_built：建筑年份
* yr\_renovated：翻修年份
* zipcode：邮政编码
* lat：维度
* long：经度
* sqft\_living15：2015年翻修后的居住面积
* sqft\_lot15：2015年翻修后的停车场面积

一些考虑：

* 唯一id在数据库中有用，在训练时并不是一个特征，所以要去掉
* 售出日期，由于是在一年内的数据，所以也没有用
* sqft\_liging15的值，如果非0的话，应该替换掉sqft\_living
* sqft\_lot15的值，如果非0的话，应该替换掉sqft\_lot
* 邮政编码对应的地理位置过于宽泛，只能引起噪音，应该去掉
* 返修年份，笔者认为它如果是非0值的话，可以替换掉建筑年份
* 看房记录次数多并不能代表该房子价格就高，而是因为地理位置、价格、配置等满足特定人群的要求，所以笔者认为它不是必须的特征值

所以最后只留下13个字段。

### 数据处理

原始数据只有一个数据集，所以需要我们自己把它分成训练集和测试集，比例大概为4:1。此数据集为csv文件格式，为了方便，我们把它转换成了两个扩展名为npz的numpy压缩形式：

* house\_Train.npz，训练数据集
* house\_Test.npz，测试数据集

### 加载数据

与上面第一个例子的代码相似，但是房屋数据属性繁杂，所以需要做归一化，房屋价格也是至少6位数，所以也需要做归一化。

这里有个需要注意的地方，即训练集和测试集的数据，需要合并在一起做归一化，然后再分开使用。为什么要先合并呢？假设训练集样本中的房屋面积的范围为150到220，而测试集中的房屋面积有可能是160到230，两者不一致。分别归一化的话，150变成0，160也变成0，这样预测就会产生误差。

最后还需要在训练集中用GenerateValidaionSet\(k=10\)分出一个1:9的验证集。

## 搭建模型

在不知道一个问题的实际复杂度之前，我们不妨把模型设计得复杂一些。如下图所示，这个模型包含了四组全连接层-Relu层的组合，最后是一个单输出做拟合。

![&#x56FE;14-5 &#x5B8C;&#x6210;&#x623F;&#x4EF7;&#x9884;&#x6D4B;&#x4EFB;&#x52A1;&#x7684;&#x62BD;&#x8C61;&#x6A21;&#x578B;](../.gitbook/assets/image%20%28326%29.png)

```python
def model():
    dr = LoadData()

    num_input = dr.num_feature
    num_hidden1 = 32
    num_hidden2 = 16
    num_hidden3 = 8
    num_hidden4 = 4
    num_output = 1

    max_epoch = 1000
    batch_size = 16
    learning_rate = 0.1

    params = HyperParameters_4_0(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.Fitting,
        init_method=InitialMethod.Xavier,
        stopper=Stopper(StopCondition.StopDiff, 1e-7))

    net = NeuralNet_4_0(params, "HouseSingle")

    fc1 = FcLayer_1_0(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "r1")
    ......
    fc5 = FcLayer_1_0(num_hidden4, num_output, params)
    net.add_layer(fc5, "fc5")

    net.train(dr, checkpoint=10, need_test=True)
    
    output = net.inference(dr.XTest)
    real_output = dr.DeNormalizeY(output)
    mse = np.sum((dr.YTestRaw - real_output)**2)/dr.YTest.shape[0]/10000
    print("mse=", mse)
    
    net.ShowLossHistory()

    ShowResult(net, dr)
```

超参数说明：

1. 学习率=0.1
2. 最大epoch=1000
3. 批大小=16
4. 拟合网络
5. 初始化方法Xavier
6. 停止条件为相对误差1e-7

net.train\(\)函数是一个阻塞函数，只有当训练完毕后才返回。

在train后面的部分，是用测试集来测试该模型的准确度，使用了数据城堡\(Data Castle\)的官方评测方法，用均方差除以10000，得到的数字越小越好。一般的模型大概是一个7位数的结果，稍微好一些的是6位数。

## 训练结果

![&#x56FE;14-6 &#x8BAD;&#x7EC3;&#x8FC7;&#x7A0B;&#x4E2D;&#x635F;&#x5931;&#x51FD;&#x6570;&#x503C;&#x548C;&#x51C6;&#x786E;&#x7387;&#x7684;&#x53D8;&#x5316;](../.gitbook/assets/image%20%28321%29.png)

由于标签数据也做了归一化，变换为都是0至1间的小数，所以均方差的数值很小，需要观察小数点以后的第4位。从图14-6中可以看到，损失函数值很快就降到了0.0002以下，然后就很缓慢地下降。而精度值在不断的上升，相信更多的迭代次数会带来更高的精度。

再看下面的打印输出部分，用R2\_Score法得到的值为0.841，而用数据城堡官方的评测标准，得到的MSE值为2384411，还比较大，说明模型精度还应该有上升的空间。

```python
......
epoch=999, total_iteration=972999
loss_train=0.000079, accuracy_train=0.740406
loss_valid=0.000193, accuracy_valid=0.857289
time used: 193.5549156665802
testing...
0.8412989144927305
mse= 2384411.5840510926
```

## 代码位置

原代码位置：[ch14, Level2](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch14-DnnBasic/Level2_HousePriceRegression.py)

个人代码：[**HousePriceRegression**](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/DNN/HousePriceRegression.py)\*\*\*\*

## keras实现

```python
from MiniFramework.DataReader_2_0 import *

from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_data():
    train_file = "../data/ch14.house.train.npz"
    test_file = "../data/ch14.house.test.npz"

    dataReader = DataReader_2_0(train_file, test_file)
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.NormalizeY(NetType.Fitting)
    dataReader.Shuffle()
    dataReader.GenerateValidationSet(k=10)
    return dataReader


def gen_data(dataReader):
    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    x_test, y_test = dataReader.XTest, dataReader.YTest
    x_val, y_val = dataReader.XDev, dataReader.YDev

    return x_train, y_train, x_test, y_test, x_val, y_val


def showResult(net, dr):
    y_test_result = net.predict(dr.XTest[0:1000, :])
    y_test_real = dr.DeNormalizeY(y_test_result)
    plt.scatter(y_test_real, y_test_real - dr.YTestRaw[0:1000, :], marker='o', label='test data')
    plt.show()


def build_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(13, )))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='Adam',
                  loss='mean_squared_error')
    return model

#画出训练过程中训练和验证的精度与损失
def draw_train_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    dataReader = load_data()
    x_train, y_train, x_test, y_test, x_val, y_val = gen_data(dataReader)

    model = build_model()
    history = model.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x_val, y_val))
    draw_train_history(history)
    showResult(model, dataReader)

    loss = model.evaluate(x_test, y_test)
    print("test loss: {}".format(loss))

    weights = model.get_weights()
    print("weights: ", weights)
```

模型输出

```python
test loss: 0.0003824683979731346
weights:  [array([[-7.80718327e-02, -3.10383588e-01,  4.49726075e-01,
...
[-0.21321435],
       [ 0.43942693],
       [-0.88441443]], dtype=float32), array([0.01937102], dtype=float32)]
```

模型损失曲线

![](../.gitbook/assets/image%20%28317%29.png)

