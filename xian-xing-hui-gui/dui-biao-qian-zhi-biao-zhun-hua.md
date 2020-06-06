# 对标签值标准化

## 发现问题

这一节里我们重点解决在训练过程中的数值的数量级的问题。

我们既然已经对样本数据特征值做了标准化，那么如此大数值的损失函数值是怎么来的呢？看一看损失函数定义：

$$ J(w,b)=\frac{1}{2m} \sum_{i=1}^m (z_i-y_i)^2 \tag{1} $$

其中，$$z_i$$是预测值，$$y_i$$是标签值。初始状态时，W和B都是0，所以，经过前向计算函数$$Z=X \cdot W+B$$的结果是0，但是Y值很大，处于\[181.38, 674.37\]之间，再经过平方计算后，一下子就成为至少5位数的数值了。

再看反向传播时的过程：

```python
def __backwardBatch(self, batch_x, batch_y, batch_z):
    m = batch_x.shape[0]
    dZ = batch_z - batch_y
    dB = dZ.sum(axis=0, keepdims=True)/m
    dW = np.dot(batch_x.T, dZ)/m
    return dW, dB
```

第二行代码求得的dZ，与房价是同一数量级的，这样经过反向传播后，dW和dB的值也会很大，导致整个反向传播链的数值都很大。我们可以debug一下，得到第一反向传播时的数值是：

```text
dW
array([[-142.59982906],
       [-283.62409678]])
dB
array([[-443.04543906]])
```

上述数值又可能在读者的机器上是不一样的，因为样本做了shuffle，但是不影响我们对问题的分析。

这么大的数值，需要把学习率设置得很小，比如0.001，才可以落到\[0,1\]区间，但是损失函数值还是不能变得很小。

如果我们像对特征值做标准化一样，把标签值也标准化到\[0,1\]之间，是不是有帮助呢？

## 代码实现

参照X的标准化方法，对Y的标准化公式如下：

$$y_{new} = \frac{y-y_{min}}{y_{max}-y_{min}} \tag{2}$$

在SimpleDataReader类中增加新方法如下：

```python
class SimpleDataReader(object):
    def NormalizeY(self):
        self.Y_norm = np.zeros((1,2))
        max_value = np.max(self.YRaw)
        min_value = np.min(self.YRaw)
        # min value
        self.Y_norm[0, 0] = min_value 
        # range value
        self.Y_norm[0, 1] = max_value - min_value 
        y_new = (self.YRaw - min_value) / self.Y_norm[0, 1]
        self.YTrain = y_new
```

原始数据中，Y的数值范围是：

* 最大值：674.37
* 最小值：181.38
* 平均值：420.64

标准化后，Y的数值范围是：

* 最大值：1.0
* 最小值：0.0
* 平均值：0.485

注意，我们同样记住了Y\_norm的值便于以后使用。

修改主程序代码，增加对Y标准化的方法调用NormalizeY\(\)：

```python
# main
if __name__ == '__main__':
    # data
    reader = SimpleDataReader()
    reader.ReadData()
    reader.NormalizeX()
    reader.NormalizeY()
    ......
```

## 运行结果

运行上述代码得到的结果其实并不令人满意：

```text
......
199 99 0.0015663978030319194 
[[-0.08194777] [ 0.80973365]] [[0.12714971]]
W= [[-0.08194777]
 [ 0.80973365]]
B= [[0.12714971]]
z= [[0.61707273]]
```

虽然W和B的值都已经处于\[-1,1\]之间了，但是z的值也在\[0,1\]之间，一套房子不可能卖0.61万元！

聪明的读者可能会想到：既然对标签值做了标准化，那么我们在得到预测结果后，需要对这个结果应该做反标准化。

根据公式2，反标准化的公式应该是：

$$
y=y_{n e w} *\left(y_{\max }-y_{\min }\right)+y_{\min } \tag{3}
$$

代码如下：

```python
if __name__ == '__main__':
    # data
    reader = SimpleDataReader()
    reader.ReadData()
    reader.NormalizeX()
    reader.NormalizeY()
    # net
    params = HyperParameters(eta=0.01, max_epoch=200, batch_size=10, eps=1e-5)
    net = NeuralNet(params, 2, 1)
    net.train(reader, checkpoint=0.1)
    # inference
    x1 = 15
    x2 = 93
    x = np.array([x1,x2]).reshape(1,2)
    x_new = reader.NormalizePredicateData(x)
    z = net.inference(x_new)
    print("z=", z)
    Z_real = z * reader.Y_norm[0,1] + reader.Y_norm[0,0]
    print("Z_real=", Z_real)
```

倒数第二行代码，就是公式3。运行结果如下：

```python
W= [[-0.08149004]
 [ 0.81022449]]
B= [[0.12801985]]
z= [[0.61856996]]
Z_real= [[486.33591769]]
```

看Z\_real的值，完全满足要求！

**总结一下从本章中学到的正确的方法：**

1. X必须标准化，否则无法训练；
2. Y值不在\[0,1\]之间时，要做标准化，好处是迭代次数少；
3. 如果Y做了标准化，对得出来的预测结果做关于Y的反标准化

至此，我们完美地解决了北京通州地区的房价预测问题！

## 总结

归纳总结一下前面遇到的困难及解决办法：

1. 样本不做标准化的话，网络发散，训练无法进行；
2. 训练样本标准化后，网络训练可以得到结果，但是预测结果有问题；
3. 还原参数值后，预测结果正确，但是此还原方法并不能普遍适用；
4. 标准化测试样本，而不需要还原参数值，可以保证普遍适用；
5. 标准化标签值，可以使得网络训练收敛快，但是在预测时需要把结果反标准化，以便得到真实值。

## 代码位置

原代码位置：[ch05, Level6](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch05-MultiVariableLinearRegression/level6_NormalizeLabelData.py)

个人代码：[**NormalizeLabelData**](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/MultiVariableLinearRegression/NormalizeLabelData.py)\*\*\*\*

## **keras实现**

```python
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler

from HelperClass.DataReader_1_0 import *

import matplotlib.pyplot as plt


def get_data():
    sdr = DataReader_1_0("../data/ch05.npz")
    sdr.ReadData()
    X,Y = sdr.GetWholeTrainSamples()
    x_mean = np.mean(X)
    x_std = np.std(X)
    y_mean = np.mean(Y)
    y_std = np.std(Y)
    ss = StandardScaler()
    # 对训练样本做归一化
    X = ss.fit_transform(X)
    # 对训练标签做归一化
    Y = ss.fit_transform(Y)

    # test data
    x1 = 15
    x2 = 93
    x = np.array([x1, x2]).reshape(1, 2)
    # 对测试数据做归一化
    x_new = NormalizePredicateData(x, x_mean, x_std)

    return X, Y, x_new, y_mean, y_std


# 手动进行归一化过程
def NormalizePredicateData(X_raw, x_mean, x_std):
    X_new = np.zeros(X_raw.shape)
    n = X_raw.shape[1]
    for i in range(n):
        col_i = X_raw[:,i]
        X_new[:,i] = (col_i - x_mean) / x_std
    return X_new


def build_model():
    model = Sequential()
    model.add(Dense(1, activation='linear', input_shape=(2,)))
    model.compile(optimizer='SGD',
                  loss='mse')
    return model


# 绘制loss曲线
def plt_loss(history):
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.title('Training  loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    X, Y, x_new, y_mean, y_std = get_data()
    x = np.array(X)
    y = np.array(Y)
    print(x.shape)
    print(y.shape)
    print(x)

    model = build_model()
    # patience设置当发现loss没有下降的情况下，经过patience个epoch后停止训练
    early_stopping = EarlyStopping(monitor='loss', patience=100)
    history = model.fit(x, y, epochs=200, batch_size=10, callbacks=[early_stopping])
    w, b = model.layers[0].get_weights()
    print(w)
    print(b)
    plt_loss(history)

    # inference
    z = model.predict(x_new)
    print("z=", z)
    # 将标签还原回真实值
    Z_true = z * y_std + y_mean
    print("Z_true=", Z_true)
```

