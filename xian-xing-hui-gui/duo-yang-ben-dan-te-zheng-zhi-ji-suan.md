# 多样本单特征值计算

在前面的代码中，我们一直使用单样本计算来实现神经网络的训练过程，但是单样本计算有一些缺点：

1. 很有可能前后两个相邻的样本，会对反向传播产生相反的作用而互相抵消。假设样本1造成了误差为0.5，w的梯度计算结果是0.1；紧接着样本2造成的误差为-0.5，w的梯度计算结果是-0.1，那么前后两次更新w就会产生互相抵消的作用。
2. 在样本数据量大时，逐个计算会花费很长的时间。由于我们在本例中样本量不大（200个样本），所以计算速度很快，觉察不到这一点。在实际的工程实践中，动辄10万甚至100万的数据量，轮询一次要花费很长的时间。

如果使用多样本计算，就要涉及到矩阵运算了，而所有的深度学习框架，都对矩阵运算做了优化，会大幅提升运算速度。打个比方：如果200个样本，循环计算一次需要2秒的话，那么把200个样本打包成矩阵，做一次计算也许只需要0.1秒。

下面我们来看看多样本运算会对代码实现有什么影响，假设我们一次用3个样本来参与计算，每个样本只有1个特征值。

## 前向计算

由于有多个样本同时计算，所以我们使用$$x_i$$表示第 $$i$$ 个样本，X是样本组成的矩阵，Z是计算结果矩阵，w和b都是标量：

$$ Z = X \cdot w + b \tag{1} $$

把它展开成3个样本（3行，每行代表一个样本）的形式：

$$
\begin{aligned}
&X=\left(\begin{array}{l}
x_{1} \\
x_{2} \\
x_{3}
\end{array}\right)\\
&Z=\left(\begin{array}{l}
x_{1} \\
x_{2} \\
x_{3}
\end{array}\right) \cdot w+b\\
&=\left(\begin{array}{l}
x_{1} \cdot w+b \\
x_{2} \cdot w+b \\
x_{3} \cdot w+b
\end{array}\right)=\left(\begin{array}{l}
z_{1} \\
z_{2} \\
z_{3}
\end{array}\right)
\end{aligned} \tag{2}
$$

$$z_1、z_2、z_3$$是三个样本的计算结果。根据公式1和公式2，我们的前向计算python代码可以写成：

```python
    def __forwardBatch(self, batch_x):
        Z = np.dot(batch_x, self.w) + self.b
        return Z
```

Python中的矩阵乘法命名有些问题，**np.dot\(\)并不是矩阵点乘，而是矩阵叉乘**，请读者习惯。

{% hint style="info" %}
矩阵叉乘

矩阵的乘法就是矩阵A的第一行乘以矩阵B的第一列，各个元素对应相乘然后求和作为第一元素的值。矩阵只有当左边矩阵的列数等于右边矩阵的行数时,它们才可以相乘,乘积矩阵的行数等于左边矩阵的行数,乘积矩阵的列数等于右边矩阵的列数。

矩阵的点乘

就是矩阵各个对应元素相乘, 这个时候要求两个矩阵必须同样大小
{% endhint %}

## 损失函数

用传统的均方差函数，其中，z是每一次迭代的预测输出，y是样本标签数据。我们使用m个样本参与计算，因此损失函数为：

$$J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}(z_i - y_i)^2$$

其中的分母中有个2，实际上是想在求导数时把这个2约掉，没有什么原则上的区别。

我们假设**每次有3个样本参与计算**，即m=3，则损失函数实例化后的情形是：

$$
\begin{array}{c}
J(w, b)=\frac{1}{2 \times 3}\left[\left(z_{1}-y_{1}\right)^{2}+\left(z_{2}-y_{2}\right)^{2}+\left(z_{3}-y_{3}\right)^{2}\right] \\
=\operatorname{sum}\left[(Z-Y)^{2}\right] / 3 / 2 \tag{3}
\end{array}
$$

**公式3中大写的Z和Y都是矩阵形式**，用代码实现：

```python
    def __checkLoss(self, dataReader):
        X,Y = dataReader.GetWholeTrainSamples()
        m = X.shape[0]
        Z = self.__forwardBatch(X)
        LOSS = (Z - Y)**2
        loss = LOSS.sum()/m/2
        return loss
```

Python中的矩阵减法运算，不需要对矩阵中的每个对应的元素单独做减法，而是整个矩阵相减即可。做求和运算时，也不需要自己写代码做遍历每个元素，而是简单地调用求和函数即可。

## 求w的梯度

我们用 J 的值作为基准，去求 w 对它的影响，也就是 J 对 w 的偏导数，就可以得到w的梯度了。从公式3看 J 的计算过程，$$z_1、z_2、z_3$$都对它有贡献；再从公式2看$$z_1、z_2、z_3$$的生成过程，都有w的参与。所以，J对w的偏导应该是这样的：

$$
\begin{aligned}
&\frac{\partial J}{\partial w}=\frac{\partial J}{\partial z_{1}} \frac{\partial z_{1}}{\partial w}+\frac{\partial J}{\partial z_{2}} \frac{\partial z_{2}}{\partial w}+\frac{\partial J}{\partial z_{3}} \frac{\partial z_{3}}{\partial w}\\
&=\frac{1}{3}\left[\left(z_{1}-y_{1}\right) x_{1}+\left(z_{2}-y_{2}\right) x_{2}+\left(z_{3}-y_{3}\right) x_{3}\right]\\
&=\frac{1}{3}\left(\begin{array}{lll}
x_{1} & x_{2} & x_{3}
\end{array}\right)\left(\begin{array}{l}
z_{1}-y_{1} \\
z_{2}-y_{2} \\
z_{3}-y_{3}
\end{array}\right)
\end{aligned}
$$

$$
=\frac{1}{m} X^T \cdot (Z-Y) \tag{4}
$$

$$
=\frac{1}{m} \sum^m_{i=1} (z_i-y_i)x_i \tag{5}
$$

其中： 

$$
\begin{aligned}
&X=\left(\begin{array}{l}
x_{1} \\
x_{2} \\
x_{3}
\end{array}\right)\\
&X^{T}=\left(\begin{array}{lll}
x_{1} & x_{2} & x_{3}
\end{array}\right)
\end{aligned}
$$

公式4和公式5其实是等价的，只不过公式5用求和方式计算每个样本，公式4用矩阵方式做一次性计算。

## 求b的梯度

$$
\frac{\partial{J}}{\partial{b}}=\frac{\partial{J}}{\partial{z_1}}\frac{\partial{z_1}}{\partial{b}}+\frac{\partial{J}}{\partial{z_2}}\frac{\partial{z_2}}{\partial{b}}+\frac{\partial{J}}{\partial{z_3}}\frac{\partial{z_3}}{\partial{b}}
$$

$$
=\frac{1}{3}[(z_1-y_1)+(z_2-y_2)+(z_3-y_3)]
$$

$$
=\frac{1}{m}\cdot (Z-Y) \tag{6}
$$

$$
=\frac{1}{m} \sum^m_{i=1} (z_i-y_i)\tag{7}
$$

公式6和公式7也是等价的，在python中，可以直接用公式6求矩阵的和，免去了一个个计算$$z_i-y_i$$最后再求和的麻烦，速度还快。

```python
    def __backwardBatch(self, batch_x, batch_y, batch_z):
        m = batch_x.shape[0]
        dZ = batch_z - batch_y
        dW = np.dot(batch_x.T, dZ)/m
        dB = dZ.sum(axis=0, keepdims=True)/m
        return dW, dB
```

## 代码位置

原代码位置：[ch04, HelperClass/NeuralNet.py](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch04-SingleVariableLinearRegression/HelperClass/NeuralNet_1_0.py)

个人代码：[NeuralNet](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/singleVariableLinearRegression/HelperClass/NeuralNet_1_0.py)

