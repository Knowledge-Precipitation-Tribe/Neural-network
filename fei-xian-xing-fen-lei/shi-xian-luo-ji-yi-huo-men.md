# 实现逻辑异或门

## 代码实现

### 准备数据

异或数据比较简单，只有4个记录，所以就hardcode在此，不用再建立数据集了。这也给读者一个机会了解如何从DataReader类派生出一个全新的子类XOR\_DataReader。

比如在下面的代码中，我们覆盖了父类中的三个方法：

* init\(\) 初始化方法：因为父类的初始化方法要求有两个参数，代表train/test数据文件
* ReadData\(\)方法：父类方法是直接读取数据文件，此处直接在内存中生成样本数据，并且直接令训练集等于原始数据集（不需要归一化），令测试集等于训练集
* GenerateValidationSet\(\)方法，由于只有4个样本，所以直接令验证集等于训练集

因为NeuralNet2中的代码要求数据集比较全，有训练集、验证集、测试集，为了已有代码能顺利跑通，我们把验证集、测试集都设置成与训练集一致，对于解决这个异或问题没有什么影响。

```python
class XOR_DataReader(DataReader):
    def ReadData(self):
        self.XTrainRaw = np.array([0,0,0,1,1,0,1,1]).reshape(4,2)
        self.YTrainRaw = np.array([0,1,1,0]).reshape(4,1)
        self.XTrain = self.XTrainRaw
        self.YTrain = self.YTrainRaw
        self.num_category = 1
        self.num_train = self.XTrainRaw.shape[0]
        self.num_feature = self.XTrainRaw.shape[1]
        self.XTestRaw = self.XTrainRaw
        self.YTestRaw = self.YTrainRaw
        self.XTest = self.XTestRaw
        self.YTest = self.YTestRaw
        self.num_test = self.num_train

    def GenerateValidationSet(self, k = 10):
        self.XVld = self.XTrain
        self.YVld = self.YTrain
```

### 测试函数

与逻辑与门和或门一样，我们需要神经网络的运算结果达到一定的精度，也就是非常的接近0，1两端，而不是说勉强大于0.5就近似为1了，所以精度要求是误差绝对值小于1e-2。

```python
def Test(dataReader, net):
    print("testing...")
    X,Y = dataReader.GetTestSet()
    A = net.inference(X)
    diff = np.abs(A-Y)
    result = np.where(diff < 1e-2, True, False)
    if result.sum() == dataReader.num_test:
        return True
    else:
        return False
```

### 主过程代码

```python
if __name__ == '__main__':
    ......
    n_input = dataReader.num_feature
    n_hidden = 2
    n_output = 1
    eta, batch_size, max_epoch = 0.1, 1, 10000
    eps = 0.005
    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.BinaryClassifier, InitialMethod.Xavier)
    net = NeuralNet2(hp, "Xor_221")
    net.train(dataReader, 100, True)
    ......
```

此处的代码有几个需要强调的细节：

* n\_input = dataReader.num\_feature，值为2，而且必须为2，因为只有两个特征值
* n\_hidden=2，这是人为设置的隐层神经元数量，可以是大于2的任何整数
* eps精度=0.005是后验知识，笔者通过测试得到的停止条件，用于方便案例讲解
* 网络类型是NetType.BinaryClassifier，指明是二分类网络
* 最后要调用Test函数验证精度

## 运行结果

经过快速的迭代后，会显示训练过程如图10-10所示。

![&#x56FE;10-10 &#x8BAD;&#x7EC3;&#x8FC7;&#x7A0B;&#x4E2D;&#x7684;&#x635F;&#x5931;&#x51FD;&#x6570;&#x503C;&#x548C;&#x51C6;&#x786E;&#x7387;&#x503C;&#x7684;&#x53D8;&#x5316;](../.gitbook/assets/image%20%28214%29.png)

可以看到二者的走势很理想。

同时在控制台会打印一些信息，最后几行如下：

```python
......
epoch=5799, total_iteration=23199
loss_train=0.005553, accuracy_train=1.000000
loss_valid=0.005058, accuracy_valid=1.000000
epoch=5899, total_iteration=23599
loss_train=0.005438, accuracy_train=1.000000
loss_valid=0.004952, accuracy_valid=1.000000
W= [[-7.10166559  5.48008579]
 [-7.10286572  5.48050039]]
B= [[ 2.91305831 -8.48569781]]
W= [[-12.06031599]
 [-12.26898815]]
B= [[5.97067802]]
testing...
1.0
None
testing...
A2= [[0.00418973]
 [0.99457721]
 [0.99457729]
 [0.00474491]]
True
```

一共用了5900个epoch，达到了指定的loss精度（0.005），loss\_valid是0.004991，刚好小于0.005时停止迭代。

我们特意打印出了A2值，即网络推理结果，如表10-7所示。

表10-7 异或计算值与神经网络推理值的比较

| x1 | x2 | XOR | Inference | diff |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 0 | 0 | 0.0041 | 0.0041 |
| 0 | 1 | 1 | 0.9945 | 0.0055 |
| 1 | 0 | 1 | 0.9945 | 0.0055 |
| 1 | 1 | 0 | 0.0047 | 0.0047 |

表中第四列的推理值与第三列的XOR结果非常的接近，继续训练的话还可以得到更高的精度，但是一般没这个必要了。由此我们再一次认识到，神经网络只可以得到无限接近真实值的近似解。

## 代码位置

原代码位置：[ch10, Level1](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch10-NonLinearBinaryClassification/Level1_XorGateClassifier.py)

个人代码：[**XorGateClassifier**](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/NonLinearBinaryClassification/XorGateClassifier.py)\*\*\*\*

## keras实现

```python
from XorGateClassifier import *

from keras.models import Sequential
from keras.layers import Dense

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_data():
    dataReader = XOR_DataReader()
    dataReader.ReadData()
    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    return x_train, y_train


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
    x_train, y_train = load_data()

    model = build_model()
    history = model.fit(x_train, y_train, epochs=5000, batch_size=1, validation_data=(x_train, y_train))
    draw_train_history(history)

    loss, accuracy = model.evaluate(x_train, y_train)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    weights = model.get_weights()
    print("weights: ", weights)
```

程序输出

```python
test loss: 0.48245617747306824, test accuracy: 0.75
weights:  [array([[ 2.7659676, -8.135253 ],
       [ 2.790494 , -8.361388 ]], dtype=float32), array([2.6930966, 1.8859061], dtype=float32), array([[ 0.21971573],
       [-5.7732844 ]], dtype=float32), array([0.45315525], dtype=float32)]
```

损失函数以及精确率曲线

![](../.gitbook/assets/image%20%28224%29.png)

