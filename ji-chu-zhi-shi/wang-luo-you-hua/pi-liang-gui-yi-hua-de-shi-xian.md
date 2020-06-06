# 批量归一化的实现

在这一节中，我们将会动手实现一个批量归一化层，来验证批量归一化的实际作用。

## 反向传播

在上一节中，我们知道了批量归一化的正向计算过程，这一节中，为了实现完整的批量归一化层，我们首先需要推导它的反向传播公式，然后用代码实现。本节中的公式序号接上一节，以便于说明。

首先假设已知从上一层回传给批量归一化层的误差矩阵是：

$$\delta = {dJ \over dZ}，\delta_i = {dJ \over dz_i} \tag{10}$$

### 求批量归一化层参数梯度

则根据公式9，求$$\gamma$$ 和$$ \beta$$的梯度：

$${dJ \over d\gamma} = \sum_{i=1}^m {dJ \over dz_i}{dz_i \over d\gamma}=\sum_{i=1}^m \delta_i \cdot n_i \tag{11}$$

$${dJ \over d\beta} = \sum_{i=1}^m {dJ \over dz_i}{dz_i \over d\beta}=\sum_{i=1}^m \delta_i \tag{12}$$

注意$$\gamma$$和$$\beta$$的形状与批大小无关，只与特征值数量有关，我们假设特征值数量为1，所以它们都是一个标量。在从计算图看，它们都与N,Z的全集相关，而不是某一个样本，因此会用求和方式计算。

### 求批量归一化层的前传误差矩阵

下述所有乘法都是element-wise的矩阵点乘，不再特殊说明。

从正向公式中看，对z有贡献的数据链是：

* $$z_i \leftarrow n_i \leftarrow x_i$$
* $$z_i \leftarrow n_i \leftarrow \mu_B \leftarrow x_i$$
* $$z_i \leftarrow n_i \leftarrow \sigma^2_B \leftarrow x_i$$
* $$z_i \leftarrow n_i \leftarrow \sigma^2_B \leftarrow \mu_B \leftarrow x_i$$

从公式8，9：

$$ {dJ \over dx_i} = {dJ \over d n_i}{d n_i \over dx_i} + {dJ \over d \sigma^2_B}{d \sigma^2_B \over dx_i} + {dJ \over d \mu_B}{d \mu_B \over dx_i} \tag{13} $$

公式13的右侧第一部分（与全连接层形式一样）：

$$ {dJ \over d n_i}= {dJ \over dz_i}{dz_i \over dn_i} = \delta_i \cdot \gamma\tag{14} $$

上式等价于：

$$ {dJ \over d N}= \delta \cdot \gamma\tag{14} $$

公式14中，我们假设样本数为64，特征值数为10，则得到一个64x10的结果矩阵（因为1x10的矩阵会被广播为64x10的矩阵）：

$$
\delta^{(64 \times 10)} \odot \gamma^{(1 \times 10)}=R^{(64 \times 10)}
$$

公式13的右侧第二部分，从公式8： 

$$ {d n_i \over dx_i}={1 \over \sqrt{\sigma^2_B + \epsilon}} \tag{15} $$

公式13的右侧第三部分，从公式8（注意$\sigma^2\_B$是个标量，而且与X,N的全集相关，要用求和方式）：

$$ \begin{aligned} {dJ \over d \sigma^2_B} = \sum_{i=1}^m {dJ \over d n_i}{d n_i \over d \sigma^2_B} \ &= -{1 \over 2}(\sigma^2_B + \epsilon)^{-3/2}\sum_{i=1}^m {dJ \over d n_i} \cdot (x_i-\mu_B) \end{aligned} \tag{16} $$

公式13的右侧第四部分，从公式7： 

$$ {d \sigma^2_B \over dx_i} = {2(x_i - \mu_B) \over m} \tag{17} $$

公式13的右侧第五部分，从公式7，8：

$$ {dJ \over d \mu_B}={dJ \over d n_i}{d n_i \over d \mu_B} + {dJ \over d\sigma^2_B}{d \sigma^2_B \over d \mu_B} \tag{18} $$

公式18的右侧第二部分，根据公式8：

$$ {d n_i \over d \mu_B}={-1 \over \sqrt{\sigma^2_B + \epsilon}} \tag{19} $$

公式18的右侧第四部分，根据公式7（$$\sigma^2_B$$和$$\mu_B$$与全体$$x_i$$相关，所以要用求和）：

$$ {d \sigma^2_B \over d \mu_B}=-{2 \over m}\sum_{i=1}^m (x_i- \mu_B) \tag{20} $$

所以公式18是：

$$ {dJ \over d \mu_B}=-{\delta \cdot \gamma \over \sqrt{\sigma^2_B + \epsilon}} - {2 \over m}{dJ \over d \sigma^2_B}\sum_{i=1}^m (x_i- \mu_B) \tag{18} $$

公式13的右侧第六部分，从公式6：

$$ {d \mu_B \over dx_i} = {1 \over m} \tag{21} $$

所以，公式13最后是这样的：

$$ {dJ \over dx_i} = {\delta \cdot \gamma \over \sqrt{\sigma^2_B + \epsilon}} + {dJ \over d\sigma^2_B} \cdot {2(x_i - \mu_B) \over m} + {dJ \over d\mu_B} \cdot {1 \over m} \tag{13} $$

## 代码实现

### 初始化类

```python
class BnLayer(CLayer):
    def __init__(self, input_size, momentum=0.9):
        self.gamma = np.ones((1, input_size))
        self.beta = np.zeros((1, input_size))
        self.eps = 1e-5
        self.input_size = input_size
        self.output_size = input_size
        self.momentum = momentum
        self.running_mean = np.zeros((1,input_size))
        self.running_var = np.zeros((1,input_size))
```

后面三个变量，momentum、running\_mean、running\_var，是为了计算/记录历史方差均差的。

### 前向计算

```python
    def forward(self, input, train=True):
        ......
```

前向计算完全按照上一节中的公式6到公式9实现。要注意在训练/测试阶段的不同算法，用train是否为True来做分支判断。

### 反向传播

```python
    def backward(self, delta_in, flag):
        ......
```

d\_norm\_x需要多次使用，所以先计算出来备用，以增加代码性能。

公式16中有一个$$(\sigma^2_B + \epsilon)^{-3/2}$$次方，在前向计算中，我们令：

```python
self.var = np.mean(self.x_mu**2, axis=0, keepdims=True) + self.eps
self.std = np.sqrt(self.var)
```

则：

$$
self.var \times self.std = self.var \times self.var^{0.5}=self.var^{(3/2)}
$$

放在分母中就是\(-3/2\)次方了。

另外代码中有很多np.sum\(..., axis=0, keepdims=True\)，这个和全连接层中的多样本计算一个道理，都是按样本数求和，并保持维度，便于后面的矩阵运算。

### 更新参数

```python
    def update(self, learning_rate=0.1):
        self.gamma = self.gamma - self.d_gamma * learning_rate
        self.beta = self.beta - self.d_beta * learning_rate
```

更新$$\gamma$$和$$\beta$$时，我们使用0.1作为学习率。在初始化代码中，并没有给批量归一化层指定学习率，如果有需求的话，读者可以自行添加这部分逻辑。

## 批量归一化层的实际应用

首先回忆一下MNIST的图片分类网络，当时的模型如图15-15所示。

![&#x56FE;15-15 &#x7B2C;14.6&#x8282;&#x4E2D;MNIST&#x56FE;&#x7247;&#x5206;&#x7C7B;&#x7F51;&#x7EDC;](../../.gitbook/assets/image%20%28358%29.png)

当时用了6个epoch（5763个Iteration），达到了0.12的预计loss值而停止训练。我们看看使用批量归一化后的样子，如图15-16所示。

![&#x56FE;15-16 &#x4F7F;&#x7528;&#x6279;&#x91CF;&#x5F52;&#x4E00;&#x5316;&#x540E;&#x7684;MNIST&#x56FE;&#x7247;&#x5206;&#x7C7B;&#x7F51;&#x7EDC;](../../.gitbook/assets/image%20%28355%29.png)

在全连接层和激活函数之间，加入一个批量归一化层，最后的分类函数Softmax前面不能加批量归一化。

### 主程序代码

```python
if __name__ == '__main__':
    ......
    params = HyperParameters_4_1(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        stopper=Stopper(StopCondition.StopLoss, 0.12))

    net = NeuralNet_4_1(params, "MNIST")

    fc1 = FcLayer_1_1(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    bn1 = BnLayer(num_hidden1)
    net.add_layer(bn1, "bn1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "r1")
    ......
```

前后都省略了一些代码，注意上面代码片段中的bn1，就是应用了批量归一化层。

### 运行结果

为了比较，我们使用与14.6中完全一致的参数设置来训练这个有批量归一化的模型，得到如图15-17所示的结果。

![&#x56FE;15-17 &#x4F7F;&#x7528;&#x6279;&#x91CF;&#x5F52;&#x4E00;&#x5316;&#x540E;&#x7684;MNIST&#x56FE;&#x7247;&#x5206;&#x7C7B;&#x7F51;&#x7EDC;&#x8BAD;&#x7EC3;&#x7ED3;&#x679C;](../../.gitbook/assets/image%20%28342%29.png)

打印输出的最后几行如下：

```python
......
epoch=4, total_iteration=4267
loss_train=0.079916, accuracy_train=0.968750
loss_valid=0.117291, accuracy_valid=0.967667
time used: 19.44783306121826
save parameters
testing...
0.9663
```

列表15-12比较一下使用批量归一化前后的区别。

表15-12 批量归一化的作用

|  | 不使用批量归一化 | 使用批量归一化 |
| :--- | :--- | :--- |
| 停止条件 | loss &lt; 0.12 | loss &lt; 0.12 |
| 训练次数 | 6个epoch\(5763次迭代\) | 4个epoch\(4267次迭代\) |
| 花费时间 | 17秒 | 19秒 |
| 准确率 | 96.97% | 96.63% |

使用批量归一化后，迭代速度提升，但是花费时间多了2秒，这是因为批量归一化的正向和反向计算过程还是比较复杂的，需要花费一些时间，但是批量归一化确实可以帮助网络快速收敛。如果使用GPU的话，花费时间上的差异应该可以忽略。

在准确率上的差异可以忽略，由于样本误差问题和随机初始化参数的差异，会造成最后的训练结果有细微差别。

## 代码位置

原代码位置：[ch15, Level6](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch15-DnnOptimization/Level6_Mnist_BN.py)

个人代码：[**Mnist\_BN**](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/DNN/Mnist_BN.py)\*\*\*\*

## keras实现

```python
from ExtendedDataReader.MnistImageDataReader import *

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

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
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
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
test loss: 0.08176513702145312, test accuracy: 0.978600025177002
```

模型损失以及准确率曲线

![](../../.gitbook/assets/4d8a1fa5-0c70-42e6-a730-a2412b1f2490.png) 

