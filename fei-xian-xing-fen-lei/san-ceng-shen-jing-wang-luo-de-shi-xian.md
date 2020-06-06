# 三层神经网络的实现

## 定义神经网络

为了完成MNIST分类，我们需要设计一个三层神经网络结构，如图12-2所示。

![&#x56FE;12-2 &#x4E09;&#x5C42;&#x795E;&#x7ECF;&#x7F51;&#x7EDC;&#x7ED3;&#x6784;](../.gitbook/assets/image%20%28285%29.png)

### 输入层

28x28=784个特征值：

$$
X=\begin{pmatrix}
    x_1 & x_2 & ... & x_{784}
  \end{pmatrix}
$$

### 隐层1

* 权重矩阵w1形状为784x64

$$
W1=\begin{pmatrix}
    w^1_{1,1} & w^1_{1,2} & ... & w^1_{1,64} \\
    ... & ... & & ... \\
    w^1_{784,1} & w^1_{784,2} & ... & w^1_{784,64} 
  \end{pmatrix}
$$

* 偏移矩阵b1的形状为1x64

$$
B1=\begin{pmatrix}
    b^1_{1} & b^1_{2} & ... & b^1_{64}
  \end{pmatrix}
$$

* 隐层1由64个神经元构成，其结果为1x64的矩阵

$$
Z1=\begin{pmatrix}    z^1_{1} & z^1_{2} & ... & z^1_{64}  \end{pmatrix}
$$

$$
A1=\begin{pmatrix}
    a^1_{1} & a^1_{2} & ... & a^1_{64}
  \end{pmatrix}
$$

### 隐层2

* 权重矩阵w2形状为64x16

$$
W2=\begin{pmatrix}
    w^2_{1,1} & w^2_{1,2} & ... & w^2_{1,16} \\
    ... & ... & & ... \\
    w^2_{64,1} & w^2_{64,2} & ... & w^2_{64,16} 
  \end{pmatrix}
$$

* 偏移矩阵b2的形状是1x16

$$
B2=\begin{pmatrix}
    b^2_{1} & b^2_{2} & ... & b^2_{16}
  \end{pmatrix}
$$

* 隐层2由16个神经元构成

$$
Z2=\begin{pmatrix}    z^2_{1} & z^2_{2} & ... & z^2_{16}  \end{pmatrix}
$$

$$
A2=\begin{pmatrix}
    a^2_{1} & a^2_{2} & ... & a^2_{16}
  \end{pmatrix}
$$

### 输出层

* 权重矩阵w3的形状为16x10

$$
W3=\begin{pmatrix}
    w^3_{1,1} & w^3_{1,2} & ... & w^3_{1,10} \\
    ... & ... & & ... \\
    w^3_{16,1} & w^3_{16,2} & ... & w^3_{16,10} 
  \end{pmatrix}
$$

* 输出层的偏移矩阵b3的形状是1x10

$$
B3=\begin{pmatrix}
    b^3_{1}& b^3_{2} & ... & b^3_{10}
  \end{pmatrix}
$$

* 输出层有10个神经元使用Softmax函数进行分类

$$
Z3=\begin{pmatrix}    z^3_{1} & z^3_{2} & ... & z^3_{10}  \end{pmatrix}
$$

$$
A3=\begin{pmatrix}
    a^3_{1} & a^3_{2} & ... & a^3_{10}
  \end{pmatrix}
$$

## 前向计算

我们都是用大写符号的矩阵形式的公式来描述，在每个矩阵符号的右上角是其形状。

### 隐层1

$$Z1 = X \cdot W1 + B1 \tag{1}$$

$$A1 = Sigmoid(Z1) \tag{2}$$

### 隐层2

$$Z2 = A1 \cdot W2 + B2 \tag{3}$$

$$A2 = Tanh(Z2) \tag{4}$$

### 输出层

$$Z3 = A2 \cdot W3 + B3 \tag{5}$$

$$A3 = Softmax(Z3) \tag{6}$$

我们的约定是行为样本，列为一个样本的所有特征，这里是784个特征，因为图片高和宽是28x28，总共784个点，把每一个点的值做为特征向量。

两个隐层，分别定义64个神经元和16个神经元。第一个隐层用Sigmoid激活函数，第二个隐层用Tanh激活函数。

输出层10个神经元，再加上一个Softmax计算，最后有a1,a2,...a10十个输出，分别代表0-9的10个数字。

## 反向传播

和以前的两层网络没有多大区别，只不过多了一层，而且用了tanh激活函数，目的是想把更多的梯度值回传，因为tanh函数比sigmoid函数稍微好一些，比如原点对称，零点梯度值大。

### 输出层

$$dZ3 = A3-Y \tag{7}$$

$$dW3 = A2^T \cdot dZ3 \tag{8}$$

$$dB3=dZ3 \tag{9}$$

### 隐层2

$$dA2 = dZ3 \cdot W3^T \tag{10}$$

$$dZ2 = dA2 \odot (1-A2 \odot A2) \tag{11}$$

$$dW2 = A1^T \cdot dZ2 \tag{12}$$

$$dB2 = dZ2 \tag{13}$$

### 隐层1

$$dA1 = dZ2 \cdot W2^T \tag{14}$$

$$dZ1 = dA1 \odot A1 \odot (1-A1) \tag{15}$$

$$dW1 = X^T \cdot dZ1 \tag{16}$$

$$dB1 = dZ1 \tag{17}$$

## 代码实现

在HelperClass2\NeuralNet\_3\_0.py中，下面主要列出与两层网络不同的代码。

### 初始化

```python
class NeuralNet3(object):
    def __init__(self, hp, model_name):
        ...
        self.wb1 = WeightsBias(self.hp.num_input, self.hp.num_hidden1, self.hp.init_method, self.hp.eta)
        self.wb1.InitializeWeights(self.subfolder, False)
        self.wb2 = WeightsBias(self.hp.num_hidden1, self.hp.num_hidden2, self.hp.init_method, self.hp.eta)
        self.wb2.InitializeWeights(self.subfolder, False)
        self.wb3 = WeightsBias(self.hp.num_hidden2, self.hp.num_output, self.hp.init_method, self.hp.eta)
        self.wb3.InitializeWeights(self.subfolder, False)
```

初始化部分需要构造三组WeightsBias对象，请注意各组的输入输出数量，决定了矩阵的形状。

### 前向计算

```python
def forward(self, batch_x):
    # 公式1
    self.Z1 = np.dot(batch_x, self.wb1.W) + self.wb1.B
    # 公式2
    self.A1 = Sigmoid().forward(self.Z1)
    # 公式3
    self.Z2 = np.dot(self.A1, self.wb2.W) + self.wb2.B
    # 公式4
    self.A2 = Tanh().forward(self.Z2)
    # 公式5
    self.Z3 = np.dot(self.A2, self.wb3.W) + self.wb3.B
    # 公式6
    if self.hp.net_type == NetType.BinaryClassifier:
        self.A3 = Logistic().forward(self.Z3)
    elif self.hp.net_type == NetType.MultipleClassifier:
        self.A3 = Softmax().forward(self.Z3)
    else:   # NetType.Fitting
        self.A3 = self.Z3
    #end if
    self.output = self.A3
```

前向计算部分增加了一层，并且使用Tanh\(\)做为激活函数。

* 反向传播

```python
    def backward(self, batch_x, batch_y, batch_a):
        # 批量下降，需要除以样本数量，否则会造成梯度爆炸
        m = batch_x.shape[0]

        # 第三层的梯度输入 公式7
        dZ3 = self.A3 - batch_y
        # 公式8
        self.wb3.dW = np.dot(self.A2.T, dZ3)/m
        # 公式9
        self.wb3.dB = np.sum(dZ3, axis=0, keepdims=True)/m 

        # 第二层的梯度输入 公式10
        dA2 = np.dot(dZ3, self.wb3.W.T)
        # 公式11
        dZ2,_ = Tanh().backward(None, self.A2, dA2)
        # 公式12
        self.wb2.dW = np.dot(self.A1.T, dZ2)/m 
        # 公式13
        self.wb2.dB = np.sum(dZ2, axis=0, keepdims=True)/m 

        # 第一层的梯度输入 公式8
        dA1 = np.dot(dZ2, self.wb2.W.T) 
        # 第一层的dZ 公式10
        dZ1,_ = Sigmoid().backward(None, self.A1, dA1)
        # 第一层的权重和偏移 公式11
        self.wb1.dW = np.dot(batch_x.T, dZ1)/m
        self.wb1.dB = np.sum(dZ1, axis=0, keepdims=True)/m 

    def update(self):
        self.wb1.Update()
        self.wb2.Update()
        self.wb3.Update()
```

反向传播也相应地增加了一层，注意要用对应的Tanh\(\)的反向公式。梯度更新时也是三组权重值同时更新。

* 主过程

```python
if __name__ == '__main__':
    ......
    n_input = dataReader.num_feature
    n_hidden1 = 64
    n_hidden2 = 16
    n_output = dataReader.num_category
    eta = 0.2
    eps = 0.01
    batch_size = 128
    max_epoch = 40

    hp = HyperParameters3(n_input, n_hidden1, n_hidden2, n_output, eta, max_epoch, batch_size, eps, NetType.MultipleClassifier, InitialMethod.Xavier)
    net = NeuralNet3(hp, "MNIST_64_16")
    net.train(dataReader, 0.5, True)
    net.ShowTrainingTrace(xline="iteration")
```

超参配置：第一隐层64个神经元，第二隐层16个神经元，学习率0.2，批大小128，Xavier初始化，最大训练40个epoch。

## 运行结果

损失函数值和准确度值变化曲线如图12-3。

![&#x56FE;12-3 &#x8BAD;&#x7EC3;&#x8FC7;&#x7A0B;&#x4E2D;&#x635F;&#x5931;&#x51FD;&#x6570;&#x548C;&#x51C6;&#x786E;&#x5EA6;&#x7684;&#x53D8;&#x5316;](../.gitbook/assets/image%20%28261%29.png)

打印输出部分：

```python
...
epoch=38, total_iteration=16769
loss_train=0.012860, accuracy_train=1.000000
loss_valid=0.100281, accuracy_valid=0.969400
epoch=39, total_iteration=17199
loss_train=0.006867, accuracy_train=1.000000
loss_valid=0.098164, accuracy_valid=0.971000
time used: 25.697904109954834
testing...
0.9749
```

在测试集上得到的准确度为97.49%，比较理想。

## 代码位置

原代码位置：[ch12, Level1](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch12-MultipleLayerNetwork/Level1_Mnist.py)

个人代码：[**Mnist**](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/MultipleLayerNetwork/Mnist.py)\*\*\*\*

## keras实现

```python
from HelperClass2.MnistImageDataReader import *

from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_data():
    dataReader = MnistImageDataReader(mode="vector")
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.NormalizeY(NetType.MultipleClassifier, base=0)
    dataReader.Shuffle()
    dataReader.GenerateValidationSet(k=12)

    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    x_test, y_test = dataReader.XTest, dataReader.YTest
    x_val, y_val = dataReader.XDev, dataReader.YDev

    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_val = x_val.reshape(x_val.shape[0], 28 * 28)


    return x_train, y_train, x_test, y_test, x_val, y_val

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='sigmoid', input_shape=(784, )))
    model.add(Dense(16, activation='tanh'))
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
    # print(x_train.shape)
    # print(x_test.shape)
    # print(x_val.shape)

    model = build_model()
    history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_val, y_val))
    draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    weights = model.get_weights()
    print("weights: ", weights)
```

模型输出

```python
test loss: 0.0870375128265936, test accuracy: 0.9732000231742859
weights:  [array([[-0.04416763, -0.0831976 ,  0.0491818 , ...,  0.06125046,
         0.07233486, -0.04410408],
       [ 0.07459141, -0.05226186,  0.08071021, ...,  0.08242527,
         0.06246025,  0.03782112],
       [-0.03699737,  0.03489852, -0.05133549, ..., -0.03645645,
         0.02399987,  0.04298285],
       ...,
       [-0.00431713,  0.08380585, -0.04733623, ...,  0.03701955,
         0.03011012, -0.02845777],
       [-0.00056171, -0.08018902,  0.03767765, ...,  0.02203549,
         0.06077852, -0.07980245],
       [ 0.02773111, -0.04303911, -0.04637436, ..., -0.02685583,
         0.06470672,  0.05784405]], dtype=float32), array([-0.2178483 , -0.11607476,  0.19602267, -0.10554409,  0.15889166,
        0.0111553 ,  0.0459778 ,  0.02578483, -0.16595618,  0.15542805,
        0.19878796,  0.04825827,  0.02092284,  0.20629567, -0.3258229 ,
       -0.04100333, -0.48876986, -0.34121832, -0.05936243, -0.2638674 ,
       -0.3630176 ,  0.27488017,  0.14832066,  0.18897443, -0.20154464,
       -0.10364075,  0.02167635, -0.1202698 , -0.0729856 ,  0.17827582,
       -0.27772433, -0.15785314, -0.09512892, -0.17780751,  0.12545995,
       -0.37470964, -0.1368926 ,  0.19438905,  0.17369035,  0.11772744,
        0.04376872, -0.35044155, -0.07170045, -0.2448514 ,  0.07240841,
        0.21900563, -0.23189951,  0.1810177 , -0.00938644, -0.25465354,
        0.07745566, -0.38113594,  0.29605415, -0.12837484, -0.23133644,
       -0.26987815,  0.08945947, -0.19215828,  0.17282893,  0.06560444,
       -0.17767893,  0.00600048, -0.10595869, -0.24236724], dtype=float32), array([[-0.47611555,  1.3752646 , -0.44537702, ..., -0.21789575,
         0.20955114, -0.22404195],
       [-0.717282  ,  0.6443544 , -0.03040413, ...,  0.23432288,
         0.80734706, -0.11616104],
       [ 0.3727672 ,  0.3902751 , -0.5423643 , ...,  0.1596318 ,
         0.34403726,  0.26756847],
       ...,
       [ 0.75902957,  0.51384676,  0.00193709, ..., -0.14926523,
        -0.00472789,  0.11335407],
       [-0.35468188, -0.17534426, -0.18108061, ...,  0.3904252 ,
        -0.36727306, -0.34770384],
       [-0.23232298,  0.30713066,  0.24121697, ...,  0.7247326 ,
        -0.33561125,  0.22771797]], dtype=float32), array([ 5.2180300e-03, -4.0033504e-02, -4.6612613e-04, -5.1712414e-06,
        5.1713035e-02, -2.6341882e-03,  3.8296855e-03, -1.6595989e-02,
       -1.1989737e-02, -2.6948549e-02, -4.6983832e-03,  2.0878188e-02,
       -1.8096061e-02, -2.7676636e-02,  6.3784644e-03, -3.4110453e-02],
      dtype=float32), array([[-1.7872726 , -1.1390297 , -0.11241101, -1.7204556 ,  0.94124806,
         0.91426176,  1.4910319 , -1.4714569 ,  1.7660816 ,  1.0062373 ],
       [-1.1788068 ,  0.9009072 ,  0.76631045,  1.5579054 ,  0.26699844,
        -1.7070191 , -1.3286783 , -1.0706636 , -0.02754159,  1.4929996 ],
       [-1.8599827 , -0.6073915 ,  0.2459145 ,  1.5619838 , -1.5937606 ,
         1.5725828 ,  1.7357361 , -0.34339568,  0.67041034, -0.93908393],
       [-0.8985166 , -1.4676709 ,  1.3431194 , -1.0557983 ,  1.1086314 ,
        -0.31145585,  1.0051986 ,  1.6683884 , -0.7232264 , -0.9140123 ],
       [ 1.7918229 , -0.65649295, -0.8599271 , -0.6510299 , -1.7270437 ,
         0.9130261 , -0.54638654,  0.7408626 , -0.6797246 ,  2.095453  ],
       [-0.75014603,  1.4846406 ,  1.2279544 ,  0.493454  , -0.36829314,
        -0.7996344 , -0.31512922, -0.28894722, -0.87901604, -0.65079254],
       [-1.2478052 ,  0.97415954, -1.1630299 ,  0.21130799, -0.90985537,
        -0.45961902, -1.198735  ,  0.8921993 ,  1.2739301 ,  0.7495436 ],
       [ 0.86831194,  1.0688986 ,  0.58052284, -0.07010659, -1.3578825 ,
        -1.419212  ,  1.6178447 ,  1.6946836 , -0.74717474, -1.123524  ],
       [ 0.7559511 , -1.5579324 ,  0.8834075 ,  0.87146765, -0.836242  ,
        -1.0301477 ,  0.6651063 , -1.0846394 ,  1.075413  , -0.17060034],
       [-1.4662247 , -1.6338125 , -2.1709802 ,  1.805146  ,  0.08466376,
         1.4614887 , -0.8848145 ,  1.2488719 ,  0.5031886 ,  0.49643546],
       [ 0.6656325 ,  0.90600896, -1.3039637 , -0.9757171 ,  1.0395771 ,
        -1.8879046 ,  0.5100971 , -1.2018363 ,  2.1261597 ,  0.6078376 ],
       [ 0.7361877 ,  1.2462751 , -0.35901597, -1.9295683 , -0.7405739 ,
         1.0305265 ,  1.3333114 , -1.0774715 ,  1.6389471 , -1.2660741 ],
       [-0.6941822 ,  1.1846787 ,  0.03872936,  0.58116376, -0.75004554,
         0.15604207, -1.4264783 ,  0.6287473 ,  1.1745315 , -1.2336555 ],
       [ 1.16753   , -0.76026106, -1.8622901 ,  1.1353645 , -1.319765  ,
         0.73333555,  0.97835153, -1.4179667 , -0.6634493 ,  1.4363192 ],
       [ 0.06947716, -0.8516638 ,  1.4862456 , -0.5567631 , -1.6972367 ,
        -1.8026927 , -1.215408  ,  0.49902138,  1.2807224 ,  1.7290442 ],
       [ 0.78108454, -1.4142572 ,  0.56793576,  1.0484227 , -1.0936918 ,
         1.3887827 , -1.0125997 , -0.8263526 ,  1.274888  , -0.7541467 ]],
      dtype=float32), array([-0.22374679,  0.08763656,  0.35643348, -0.06640335, -0.02601829,
        0.19632392, -0.49040654,  0.0619854 , -0.13785903,  0.0957261 ],
      dtype=float32)]
```

模型损失与准确率曲线

![](../.gitbook/assets/image%20%28274%29.png)

