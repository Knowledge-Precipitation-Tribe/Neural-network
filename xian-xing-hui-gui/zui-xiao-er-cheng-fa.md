# 最小二乘法

## 历史

最小二乘法，也叫做最小平方法（Least Square），它**通过最小化误差的平方和寻找数据的最佳函数匹配**。利用最小二乘法可以简便地求得未知的数据，并使得这些求得的数据与实际数据之间误差的平方和为最小。最小二乘法还可用于曲线拟合。其他一些优化问题也可通过最小化能量或最小二乘法来表达。

1801年，意大利天文学家朱赛普·皮亚齐发现了第一颗小行星谷神星。经过40天的跟踪观测后，由于谷神星运行至太阳背后，使得皮亚齐失去了谷神星的位置。随后全世界的科学家利用皮亚齐的观测数据开始寻找谷神星，但是根据大多数人计算的结果来寻找谷神星都没有结果。时年24岁的高斯也计算了谷神星的轨道。奥地利天文学家海因里希·奥尔伯斯根据高斯计算出来的轨道重新发现了谷神星。

高斯使用的最小二乘法的方法发表于1809年他的著作《天体运动论》中。法国科学家勒让德于1806年独立发明“最小二乘法”，但因不为世人所知而默默无闻。勒让德曾与高斯为谁最早创立最小二乘法原理发生争执。

1829年，高斯提供了最小二乘法的优化效果强于其他方法的证明，因此被称为高斯-马尔可夫定理。

## 数学原理

线性回归试图学得：

$$z(x_i)=w \cdot x_i+b \tag{1}$$

使得：

$$z(x_i) \simeq y_i \tag{2}$$

其中，$$x_i$$是样本特征值，$$y_i$$是样本标签值，$$z_i$$是模型预测值。

如何学得w和b呢？均方差\(MSE - mean squared error\)是回归任务中常用的手段： 

$$ J = \sum_{i=1}^m(z(x_i)-y_i)^2 = \sum_{i=1}^m(y_i-wx_i-b)^2 \tag{3} $$

$$J$$称为损失函数。实际上就是试图找到一条直线，使所有样本到直线上的残差的平方和最小。

![&#x56FE;4-3 &#x5747;&#x65B9;&#x5DEE;&#x51FD;&#x6570;&#x7684;&#x8BC4;&#x4F30;&#x539F;&#x7406;](../.gitbook/assets/image%20%2816%29.png)

图4-3中，圆形点是样本点，直线是当前的拟合结果。如左图所示，我们是要计算样本点到直线的垂直距离，需要再根据直线的斜率来求垂足然后再计算距离，这样计算起来很慢；但实际上，**在工程上我们通常使用的是右图的方式**，即样本点到直线的竖直距离，因为这样计算很方便，用一个减法就可以了。

假设我们计算出初步的结果是虚线所示，这条直线是否合适呢？我们来计算一下图中每个点到这条直线的距离，把这些距离的值都加起来（都是正数，不存在互相抵消的问题）成为误差。

因为上图中的几个点不在一条直线上，所以不能有一条直线能同时穿过它们。所以，我们只能想办法不断改变红色直线的角度和位置，**让总体误差最小**（永远不可能是0），就意味着整体偏差最小，那么最终的那条直线就是我们要的结果。

如果想让误差的值最小，通过对w和b求导，再令导数为0（到达最小极值），就是w和b的最优解。

推导过程如下：

$$ \begin{aligned} {\partial{J} \over \partial{w}} &={\partial{(\sum_{i=1}^m(y_i-wx_i-b)^2)} \over \partial{w}} \ = 2\sum_{i=1}^m(y_i-wx_i-b)(-x_i) \end{aligned} \tag{4} $$

令公式4为0：

$$ \sum_{i=1}^m(y_i-wx_i-b)x_i=0 \tag{5} $$

$$ \begin{aligned} {\partial{J} \over \partial{b}} &={\partial{(\sum_{i=1}^m(y_i-wx_i-b)^2)} \over \partial{b}} \ =2\sum_{i=1}^m(y_i-wx_i-b)(-1) \end{aligned} \tag{6} $$

令公式6为0：

$$ \sum_{i=1}^m(y_i-wx_i-b)=0 \tag{7} $$

由式7得到（假设有m个样本）：

$$ \sum_{i=1}^m b = m \cdot b = \sum_{i=1}^m{y_i} - w\sum_{i=1}^m{x_i} \tag{8} $$

两边除以m：

$$ b = {1 \over m}(\sum_{i=1}^m{y_i} - w\sum_{i=1}^m{x_i})=\bar y-w \bar x \tag{9} $$

其中：

$$ \bar y = {1 \over m}\sum_{i=1}^m y_i, \bar x={1 \over m}\sum_{i=1}^m x_i \tag{10} $$

将公式10代入公式5：

$$ \sum_{i=1}^m(y_i-wx_i-\bar y + w \bar x)x_i=0 $$

$$ \sum_{i=1}^m(x_i y_i-wx^2_i-x_i \bar y + w \bar x x_i)=0 $$

$$ \sum_{i=1}^m(x_iy_i-x_i \bar y)-w\sum_{i=1}^m(x^2_i - \bar x x_i) = 0 $$

$$ w = {\sum_{i=1}^m(x_iy_i-x_i \bar y) \over \sum_{i=1}^m(x^2_i - \bar x x_i)} \tag{11} $$

将公式10代入公式11：

$$ w={\sum_{i=1}^m (x_i \cdot y_i) - \sum_{i=1}^m x_i \cdot {1 \over m} \sum_{i=1}^m y_i \over \sum_{i=1}^m x^2_i - \sum_{i=1}^m x_i \cdot {1 \over m}\sum_{i=1}^m x_i} \tag{12} $$

分子分母都乘以m：

$$ w={m\sum_{i=1}^m x_i y_i - \sum_{i=1}^m x_i \sum_{i=1}^m y_i \over m\sum_{i=1}^m x^2_i - (\sum_{i=1}^m x_i)^2} \tag{13} $$

$$ b=\frac{1}{m}\sum_{i=1}^m(y_i-wx_i) \tag{14} $$

而事实上，式13有很多个变种，大家会在不同的文章里看到不同版本，往往感到困惑，比如下面两个公式也是正确的解：

$$ w = {\sum_{i=1}^m y_i(x_i-\bar x) \over \sum_{i=1}^m x^2_i - (\sum_{i=1}^m x_i)^2/m} \tag{15} $$

$$ w = {\sum_{i=1}^m x_i(y_i-\bar y) \over \sum_{i=1}^m x^2_i - \bar x \sum_{i=1}^m x_i} \tag{16} $$

以上两个公式，如果把公式10代入，也应该可以得到和式13相同的答案，只不过需要一些运算技巧。比如，很多人不知道这个神奇的公式：

$$
\begin{array}{l}
\sum\left(x_{i} \bar{y}\right)=\bar{y} \sum x_{i}=\frac{1}{m}\left(\sum y_{i}\right)\left(\sum x_{i}\right) \\
=\frac{1}{m}\left(\sum x_{i}\right)\left(\sum y_{i}\right)=\bar{x} \sum y_{i}=\sum\left(y_{i} \bar{x}\right)
\end{array} \tag{17}
$$

## 代码实现

我们下面用Python代码来实现一下以上的计算过程：

### 计算w值

```python
# 根据公式15
def method1(X,Y,m):
    x_mean = X.mean()
    p = sum(Y*(X-x_mean))
    q = sum(X*X) - sum(X)*sum(X)/m
    w = p/q
    return w

# 根据公式16
def method2(X,Y,m):
    x_mean = X.mean()
    y_mean = Y.mean()
    p = sum(X*(Y-y_mean))
    q = sum(X*X) - x_mean*sum(X)
    w = p/q
    return w

# 根据公式13
def method3(X,Y,m):
    p = m*sum(X*Y) - sum(X)*sum(Y)
    q = m*sum(X*X) - sum(X)*sum(X)
    w = p/q
    return w
```

由于有函数库的帮助，我们不需要手动计算sum\(\), mean\(\)这样的基本函数。

### 计算b值

```python
# 根据公式14
def calculate_b_1(X,Y,w,m):
    b = sum(Y-w*X)/m
    return b

# 根据公式9
def calculate_b_2(X,Y,w):
    b = Y.mean() - w * X.mean()
    return b
```

## 运算结果

用以上几种方法，最后得出的结果都是一致的，可以起到交叉验证的作用：

```text
w1=2.056827, b1=2.965434
w2=2.056827, b2=2.965434
w3=2.056827, b3=2.965434
```

## 代码位置

原代码位置：[ch04, Level1](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch04-SingleVariableLinearRegression/Level1_LeastSquare.py)

个人代码：[LeastSquare](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/singleVariableLinearRegression/LeastSquare.py)

