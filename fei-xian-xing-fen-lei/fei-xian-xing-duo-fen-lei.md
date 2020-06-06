# 非线性多分类

## 定义神经网络结构

先设计出能完成非线性多分类的网络结构，如图11-2所示。

![&#x56FE;11-2 &#x975E;&#x7EBF;&#x6027;&#x591A;&#x5206;&#x7C7B;&#x7684;&#x795E;&#x7ECF;&#x7F51;&#x7EDC;&#x7ED3;&#x6784;&#x56FE;](../.gitbook/assets/image%20%28263%29.png)

* 输入层两个特征值$$x_1, x_2$$ 

$$
x=
\begin{pmatrix}
    x_1 & x_2
\end{pmatrix}
$$

* 隐层2x3的权重矩阵$$W1 $$

$$
W1=
\begin{pmatrix}
    w^1_{11} & w^1_{12} & w^1_{13} \\
    w^1_{21} & w^1_{22} & w^1_{23}
\end{pmatrix}
$$

* 隐层1x3的偏移矩阵$$B1$$

$$
B1=\begin{pmatrix}
    b^1_1 & b^1_2 & b^1_3 
\end{pmatrix}
$$

* 隐层由3个神经元构成
* 输出层3x3的权重矩阵$$W2 $$

$$
W2=\begin{pmatrix}
    w^2_{11} & w^2_{12} & w^2_{13} \\
    w^2_{21} & w^2_{22} & w^2_{23} \\
    w^2_{31} & w^2_{32} & w^2_{33} 
\end{pmatrix}
$$

* 输出层1x1的偏移矩阵$$B2$$

$$
B2=\begin{pmatrix}
    b^2_1 & b^2_2 & b^2_3 
  \end{pmatrix}
$$

* 输出层有3个神经元使用Softmax函数进行分类

## 前向计算

根据网络结构，可以绘制前向计算图，如图11-3所示。

![&#x56FE;11-3 &#x524D;&#x5411;&#x8BA1;&#x7B97;&#x56FE;](../.gitbook/assets/image%20%28273%29.png)

### 第一层

* 线性计算

$$
z^1_1 = x_1 w^1_{11} + x_2 w^1_{21} + b^1_1
$$

$$
z^1_2 = x_1 w^1_{12} + x_2 w^1_{22} + b^1_2
$$

$$
z^1_3 = x_1 w^1_{13} + x_2 w^1_{23} + b^1_3
$$

$$
Z1 = X \cdot W1 + B1
$$

* 激活函数

$$
a^1_1 = Sigmoid(z^1_1)
$$

$$
a^1_2 = Sigmoid(z^1_2)
$$

$$
a^1_3 = Sigmoid(z^1_3)
$$

$$
A1 = Sigmoid(Z1)
$$

### 第二层

* 线性计算

$$
z^2_1 = a^1_1 w^2_{11} + a^1_2 w^2_{21} + a^1_3 w^2_{31} + b^2_1
$$

$$
z^2_2 = a^1_1 w^2_{12} + a^1_2 w^2_{22} + a^1_3 w^2_{32} + b^2_2
$$

$$
z^2_3 = a^1_1 w^2_{13} + a^1_2 w^2_{23} + a^1_3 w^2_{33} + b^2_3
$$

$$
Z2 = A1 \cdot W2 + B2
$$

* 分类函数

$$
a^2_1 = {e^{z^2_1} \over e^{z^2_1} + e^{z^2_2} + e^{z^2_3}}
$$

$$
a^2_2 = {e^{z^2_2} \over e^{z^2_1} + e^{z^2_2} + e^{z^2_3}}
$$

$$
a^2_3 = {e^{z^2_3} \over e^{z^2_1} + e^{z^2_2} + e^{z^2_3}}
$$

$$
A2 = Softmax(Z2)
$$

### 损失函数

使用多分类交叉熵损失函数： 

$$
loss = -(y_1 \ln a^2_1 + y_2 \ln a^2_2 + y_3 \ln a^2_3)
$$

$$
J(w,b) = -{1 \over m} \sum^m_{i=1} \sum^n_{j=1} y_{ij} \ln (a^2_{ij})
$$

m为样本数，n为类别数。

## 反向传播

根据前向计算图，可以绘制出反向传播的路径如图11-4。

![&#x56FE;11-4 &#x53CD;&#x5411;&#x4F20;&#x64AD;&#x56FE;](../.gitbook/assets/image%20%28269%29.png)

在之前已经学习过了Softmax与多分类交叉熵配合时的反向传播推导过程，最后是一个很简单的减法：

$$ {\partial loss \over \partial Z2}=A2-y \rightarrow dZ2 $$

从Z2开始再向前推的话，和之前推导是一模一样的，所以直接把结论拿过来：

$$ {\partial loss \over \partial W2}=A1^T \cdot dZ2 \rightarrow dW2 $$ $${\partial{loss} \over \partial{B2}}=dZ2 \rightarrow dB2$$ $$ {\partial A1 \over \partial Z1}=A1 \odot \(1-A1\) \rightarrow dA1 $$ $$ {\partial loss \over \partial Z1}=dZ2 \cdot W2^T \odot dA1 \rightarrow dZ1 $$ $$ dW1=X^T \cdot dZ1 $$ $$ dB1=dZ1 $$

## 代码实现

绝大部分代码都在HelperClass2目录中的基本类实现，这里只有主过程：

```python
if __name__ == '__main__':
    ......
    n_input = dataReader.num_feature
    n_hidden = 3
    n_output = dataReader.num_category
    eta, batch_size, max_epoch = 0.1, 10, 5000
    eps = 0.1
    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.MultipleClassifier, InitialMethod.Xavier)
    # create net and train
    net = NeuralNet2(hp, "Bank_233")
    net.train(dataReader, 100, True)
    net.ShowTrainingTrace()
    # show result
    ......
```

过程描述：

1. 读取数据文件
2. 显示原始数据样本分布图
3. 其它数据操作：归一化、打乱顺序、建立验证集
4. 设置超参
5. 建立神经网络开始训练
6. 显示训练结果

## 运行结果

训练过程如图11-5所示。

![&#x56FE;11-5 &#x8BAD;&#x7EC3;&#x8FC7;&#x7A0B;&#x4E2D;&#x7684;&#x635F;&#x5931;&#x51FD;&#x6570;&#x503C;&#x548C;&#x51C6;&#x786E;&#x7387;&#x503C;&#x7684;&#x53D8;&#x5316;](../.gitbook/assets/image%20%28277%29.png)

迭代了5000次，没有到达损失函数小于0.1的条件。

分类结果如图11-6所示。

![&#x56FE;11-6 &#x5206;&#x7C7B;&#x6548;&#x679C;&#x56FE;](../.gitbook/assets/image%20%28299%29.png)

因为没达到精度要求，所以分类效果一般。从分类结果图上看，外圈圆形差不多拟合住了，但是内圈的方形还差很多。

打印输出：

```python
......
epoch=4999, total_iteration=449999
loss_train=0.225935, accuracy_train=0.800000
loss_valid=0.137970, accuracy_valid=0.960000
W= [[ -8.30315494   9.98115605   0.97148346]
 [ -5.84460922  -4.09908698 -11.18484376]]
B= [[ 4.85763475 -5.61827538  7.94815347]]
W= [[-32.28586038  -8.60177788  41.51614172]
 [-33.68897413  -7.93266621  42.09333288]
 [ 34.16449693   7.93537692 -41.19340947]]
B= [[-11.11937314   3.45172617   7.66764697]]
testing...
0.952
```

最后的测试分类准确率为0.952。

## 代码位置

原代码位置：[ch11, Level1](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch11-NonLinearMultipleClassification/Level1_BankClassifier.py)

个人代码：[**BankClassifier**](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/NonLinearMultipleClassification/BankClassifier.py)\*\*\*\*

## keras实现

```python
from HelperClass2.DataReader_2_0 import *

from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_data():
    train_data_name = "../data/ch11.train.npz"
    test_data_name = "../data/ch11.test.npz"
    dataReader = DataReader_2_0(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.NormalizeY(NetType.MultipleClassifier, base=1)
    dataReader.NormalizeX()
    dataReader.Shuffle()
    dataReader.GenerateValidationSet()

    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    x_test, y_test = dataReader.XTest, dataReader.YTest
    x_val, y_val = dataReader.XDev, dataReader.YDev
    return x_train, y_train, x_test, y_test, x_val, y_val

def build_model():
    model = Sequential()
    model.add(Dense(3, activation='sigmoid', input_shape=(2,)))
    model.add(Dense(3, activation='softmax'))
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
    history = model.fit(x_train, y_train, epochs=200, batch_size=10, validation_data=(x_test, y_test))
    draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    weights = model.get_weights()
    print("weights: ", weights)
```

模型输出

```python
test loss: 0.5864107594490051, test accuracy: 0.6593332886695862
weights:  [array([[-3.0283191 ,  0.84802836, -0.312435  ],
       [ 0.5548505 , -7.2905626 , -0.39118102]], dtype=float32), array([-0.36451575, -0.3283058 ,  0.04513721], dtype=float32), array([[-0.04179221, -0.90641433,  0.892557  ],
       [-2.9173682 , -1.2463648 ,  1.9722716 ],
       [-0.76959646,  0.82336   ,  0.17536448]], dtype=float32), array([-0.12553275,  0.34421512, -0.30111837], dtype=float32)]
```

模型损失以及准确率曲线

![](../.gitbook/assets/image%20%28290%29.png)



