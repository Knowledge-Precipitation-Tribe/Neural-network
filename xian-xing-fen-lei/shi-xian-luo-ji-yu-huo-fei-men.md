# 实现逻辑与或非门

单层神经网络，又叫做感知机，它可以轻松实现逻辑与、或、非门。由于逻辑与、或门，需要有两个变量输入，而逻辑非门只有一个变量输入。但是它们共同的特点是输入为0或1，可以看作是正负两个类别。

所以，在学习了二分类知识后，我们可以用分类的思想来实现下列5个逻辑门：

* 与门 AND
* 与非门 NAND
* 或门 OR
* 或非门 NOR
* 非门 NOT

以逻辑AND为例，图6-12中的4个点，分别是4个样本数据，蓝色圆点表示负例（y=0），红色三角表示正例（y=1）。

![&#x56FE;6-12 &#x53EF;&#x4EE5;&#x89E3;&#x51B3;&#x903B;&#x8F91;&#x4E0E;&#x95EE;&#x9898;&#x7684;&#x591A;&#x6761;&#x5206;&#x5272;&#x7EBF;](../.gitbook/assets/image%20%28124%29.png)

如果用分类思想的话，根据前面学到的知识，应该在红色点和蓝色点之间划出一条分割线来，可以正好把正例和负例完全分开。由于样本数据稀疏，所以这条分割线的角度和位置可以比较自由，比如图中的三条直线，都可以是这个问题的解。让我们一起看看神经网络能否给我们带来惊喜。

## 实现逻辑非门

很多阅读材料上会这样介绍：有公式 $$y=wx+b$$，令$$w=-1,b=1$$，则：

* 当$$x=0$$时，$$y = -1 \times 0 + 1 = 1$$
* 当$$x=1$$时，$$y = -1 \times 1 + 1 = 0$$

于是有如图6-13所示的神经元结构。

![&#x56FE;6-13 &#x4E0D;&#x6B63;&#x786E;&#x7684;&#x903B;&#x8F91;&#x975E;&#x95E8;&#x7684;&#x795E;&#x7ECF;&#x5143;&#x5B9E;&#x73B0;](../.gitbook/assets/image%20%2884%29.png)

但是，这变成了一个拟合问题，而不是分类问题。比如，令$$x=0.5$$，带入公式中有：

$$ y=wx+b = -1 \times 0.5 + 1 = 0.5 $$

即，当$$x=0.5$$时，$$y=0.5$$，且其结果$$x$$和$$y$$的值并没有丝毫“非”的意思。所以，应该定义如图6-14所示的神经元来解决问题，而其样本数据也很简单，如表6-6所示，一共只有两行数据。

![&#x56FE;6-14 &#x6B63;&#x786E;&#x7684;&#x903B;&#x8F91;&#x975E;&#x95E8;&#x7684;&#x795E;&#x7ECF;&#x5143;&#x5B9E;&#x73B0;](../.gitbook/assets/image%20%2895%29.png)

表6-6 逻辑非问题的样本数据

| 样本序号 | 样本值x | 标签值y |
| :--- | :--- | :--- |
| 1 | 0 | 1 |
| 2 | 1 | 0 |

建立样本数据的代码如下：

```python
def Read_Logic_NOT_Data(self):
    X = np.array([0,1]).reshape(2,1)
    Y = np.array([1,0]).reshape(2,1)
    self.XTrain = self.XRaw = X
    self.YTrain = self.YRaw = Y
    self.num_train = self.XRaw.shape[0]
```

在主程序中，令：

```python
num_input = 1
num_output = 1
```

执行训练过程，最终得到图6-16所示的分类结果和下面的打印输出结果。

```python
......
2514 1 0.0020001369266925305
2515 1 0.0019993382569061806
W= [[-12.46886021]]
B= [[6.03109791]]
[[0.99760291]
 [0.00159743]]
```

![&#x56FE;6-15 &#x903B;&#x8F91;&#x975E;&#x95E8;&#x7684;&#x5206;&#x7C7B;&#x7ED3;&#x679C;](../.gitbook/assets/image%20%28121%29.png)

从图6-15中，可以理解神经网络在左右两类样本点之间画了一条直线，来分开两类样本，该直线的方程就是打印输出中的W和B值所代表的直线：

$$ y = -12.468x + 6.031 $$

但是，为什么不是一条垂直于x轴的直线呢，而是稍微有些“歪”？

这体现了神经网络的能力的局限性，它只是“模拟”出一个结果来，而不能精确地得到完美的数学公式。这个问题的精确的数学公式是一条垂直线，相当于$$w=\infty$$，这是不可能训练得出来的。

## 实现逻辑与或门

### 神经元模型

依然使用之前的神经元模型，如图6-16。

![&#x56FE;6-16 &#x903B;&#x8F91;&#x4E0E;&#x6216;&#x95E8;&#x7684;&#x795E;&#x7ECF;&#x5143;&#x5B9E;&#x73B0;](../.gitbook/assets/image%20%28128%29.png)

因为输入特征值只有两个，输出一个二分类，所以模型和前一节的一样。

### 训练样本

每个类型的逻辑门都只有4个训练样本，如表6-7所示。

表6-7 四种逻辑门的样本和标签数据

| 样本 | x1 | x2 | 逻辑与y | 逻辑与非y | 逻辑或y | 逻辑或非y |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0 | 0 | 0 | 1 | 0 | 1 |
| 2 | 0 | 1 | 0 | 1 | 1 | 0 |
| 3 | 1 | 0 | 0 | 1 | 1 | 0 |
| 4 | 1 | 1 | 1 | 0 | 1 | 0 |

### 读取数据

```python
class LogicDataReader(SimpleDataReader):
    def Read_Logic_AND_Data(self):
        X = np.array([0,0,0,1,1,0,1,1]).reshape(4,2)
        Y = np.array([0,0,0,1]).reshape(4,1)
        self.XTrain = self.XRaw = X
        self.YTrain = self.YRaw = Y
        self.num_train = self.XRaw.shape[0]

    def Read_Logic_NAND_Data(self):
        ......

    def Read_Logic_OR_Data(self):
        ......

    def Read_Logic_NOR_Data(self):        
        ......
```

以逻辑AND为例，我们从SimpleDataReader派生出自己的类LogicDataReader，并加入特定的数据读取方法Read\_Logic\_AND\_Data\(\)，其它几个逻辑门的方法类似，在此只列出方法名称。

### 测试函数

```python
def Test(net, reader):
    X,Y = reader.GetWholeTrainSamples()
    A = net.inference(X)
    print(A)
    diff = np.abs(A-Y)
    result = np.where(diff < 1e-2, True, False)
    if result.sum() == 4:
        return True
    else:
        return False
```

我们知道了神经网络只能给出近似解，但是这个“近似”能到什么程度，是需要我们在训练时自己指定的。相应地，我们要有测试手段，比如当输入为\(1，1\)时，AND的结果是1，但是神经网络只能给出一个0.721的概率值，这是不满足精度要求的，必须让4个样本的误差都小于1e-2。

### 训练函数

```python
def train(reader, title):
    ...
    params = HyperParameters(eta=0.5, max_epoch=10000, batch_size=1, eps=2e-3, net_type=NetType.BinaryClassifier)
    num_input = 2
    num_output = 1
    net = NeuralNet(params, num_input, num_output)
    net.train(reader, checkpoint=1)
    # test
    print(Test(net, reader))
    ......
```

在超参中指定了最多10000次的epoch，0.5的学习率，停止条件为损失函数值低至2e-3时。在训练结束后，要先调用测试函数，需要返回True才能算满足要求，然后用图形显示分类结果。

### 运行结果

逻辑AND的运行结果的打印输出如下：

```python
......
epoch=4236
4236 3 0.0019998012999365928
W= [[11.75750515]
 [11.75780362]]
B= [[-17.80473354]]
[[9.96700157e-01]
 [2.35953140e-03]
 [1.85140939e-08]
 [2.35882891e-03]]
True
```

迭代了4236次，达到精度$$loss<1e-2$$。当输入$$(1,1)、(1,0)、(0,1)、(0,0)$$四种组合时，输出全都满足精度要求。

## 结果比较

把5组数据放入表6-8中做一个比较。

表6-8 五种逻辑门的结果比较

| 逻辑门 | 分类结果 | 参数值 |
| :--- | :--- | :--- |
| 非 | ![](../.gitbook/assets/image%20%28134%29.png)  | W=-12.468 B=6.031 |
| 与 | ![](../.gitbook/assets/image%20%28107%29.png)  | W1=11.757 W2=11.757 B=-17.804 |
| 与非 | ![](../.gitbook/assets/image%20%28104%29.png)  | W1=-11.763 W2=-11.763 B=17.812 |
| 或 | ![](../.gitbook/assets/image%20%2887%29.png)  | W1=11.743 W2=11.743 B=-11.738 |
| 或非 | ![](../.gitbook/assets/image%20%28100%29.png)  | W1=-11.738 W2=-11.738 B=5.409 |

我们从数值和图形可以得到两个结论：

1. W1和W2的值基本相同而且符号相同，说明分割线一定是135°斜率
2. 精度越高，则分割线的起点和终点越接近四边的中点0.5的位置

以上两点说明神经网络还是很聪明的，它会尽可能优美而鲁棒地找出那条分割线。

## 代码位置

原代码位置：[ch06, Level4](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch06-LinearBinaryClassification/Level4_LogicGates.py)

个人代码：[LogicGates](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/LinearBinaryClassification/LogicGates.py)

## 思考与练习

1. 减小max\_epoch的数值，观察神经网络的训练结果。

将max\_epoch调整为5000

![](../.gitbook/assets/image%20%28136%29.png)

    2. 为什么达到相同的精度，逻辑OR和NOR只用2000次左右的epoch，而逻辑AND和NAND却需要4000次以上？

因为逻辑OR和NOR的数据使得分类线在右上角就可以达到很好的效果，而逻辑AND和NAND在输入\[0,0\]时不满足要求，输入\[0,1\]和\[1,0\]时不满足要求，所以需要继续迭代，使得逻辑AND与NAND的训练次数更多。

下图分别是epoch=1000与epoch等于5000的情况

![epoch=1000](../.gitbook/assets/image%20%28123%29.png)

![epoch=5000](../.gitbook/assets/image%20%28115%29.png)

## keras实现

```python
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

from HelperClass.NeuralNet_1_2 import *
from HelperClass.Visualizer_1_0 import *


class LogicDataReader(DataReader_1_1):
    def __init__(self):
        pass

    def Read_Logic_NOT_Data(self):
        X = np.array([0, 1]).reshape(2, 1)
        Y = np.array([1, 0]).reshape(2, 1)
        self.XTrain = self.XRaw = X
        self.YTrain = self.YRaw = Y
        self.num_train = self.XRaw.shape[0]

    def Read_Logic_AND_Data(self):
        X = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        Y = np.array([0, 0, 0, 1]).reshape(4, 1)
        self.XTrain = self.XRaw = X
        self.YTrain = self.YRaw = Y
        self.num_train = self.XRaw.shape[0]

    def Read_Logic_NAND_Data(self):
        X = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        Y = np.array([1, 1, 1, 0]).reshape(4, 1)
        self.XTrain = self.XRaw = X
        self.YTrain = self.YRaw = Y
        self.num_train = self.XRaw.shape[0]

    def Read_Logic_OR_Data(self):
        X = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        Y = np.array([0, 1, 1, 1]).reshape(4, 1)
        self.XTrain = self.XRaw = X
        self.YTrain = self.YRaw = Y
        self.num_train = self.XRaw.shape[0]

    def Read_Logic_NOR_Data(self):
        X = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        Y = np.array([1, 0, 0, 0]).reshape(4, 1)
        self.XTrain = self.XRaw = X
        self.YTrain = self.YRaw = Y
        self.num_train = self.XRaw.shape[0]


def build_model():
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_shape=(2,)))
    model.compile(optimizer='SGD', loss='binary_crossentropy')
    return model


def draw_source_data(reader, title, show=False):
    fig = plt.figure(figsize=(5, 5))
    plt.grid()
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.title(title)
    X, Y = reader.GetWholeTrainSamples()
    if title == "Logic NOT operator":
        DrawTwoCategoryPoints(X[:, 0], np.zeros_like(X[:, 0]), Y[:, 0], title=title, show=show)
    else:
        DrawTwoCategoryPoints(X[:, 0], X[:, 1], Y[:, 0], title=title, show=show)


def draw_split_line(w, b):
    x = np.array([-0.1, 1.1])
    old_w = w
    w = -w[0,0]/old_w[1,0]
    b = -b[0]/old_w[1,0]
    y = w * x + b
    plt.plot(x, y)


if __name__ == '__main__':
    reader = LogicDataReader()
    reader.Read_Logic_AND_Data()
    x, y = reader.XTrain, reader.YTrain
    print("x", x)
    print("y", y)
    model = build_model()
    model.fit(x, y, epochs=1000, batch_size=1)
    # 获得权重
    w, b = model.layers[0].get_weights()
    print(w)
    print(b)

    draw_source_data(reader, "Logic AND operator")
    draw_split_line(w, b)
    plt.show()
```

