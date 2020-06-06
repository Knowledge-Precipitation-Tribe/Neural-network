# 挤压型激活函数

这一类函数的特点是，当输入值域的绝对值较大的时候，其输出在两端是饱和的，都具有S形的函数曲线以及压缩输入值域的作用，所以叫挤压型激活函数，又可以叫饱和型激活函数。

在英文中，通常用Sigmoid来表示，原意是S型的曲线，在数学中是指一类具有压缩作用的S型的函数，在神经网络中，有两个常用的Sigmoid函数，一个是Logistic函数，另一个是Tanh函数。下面我们分别来讲解它们。

## Logistic函数

对数几率函数（Logistic Function，简称对率函数）。

很多文字材料中通常把激活函数和分类函数混淆在一起说，有一个原因是：在二分类任务中最后一层使用的对率函数与在神经网络层与层之间连接的Sigmoid激活函数，是同样的形式。所以它既是激活函数，又是分类函数，是个特例。

对这个函数的叫法比较混乱，在本书中我们约定一下，**凡是用到“Logistic”词汇的，指的是二分类函数；而用到“Sigmoid”词汇的，指的是本激活函数。**

### 公式

$$Sigmoid(z) = \frac{1}{1 + e^{-z}} \rightarrow a \tag{1}$$

### 导数

$$Sigmoid'(z) = a(1 - a) \tag{2}$$

注意，如果是矩阵运算的话，需要在公式2中使用$$\odot$$符号表示按元素的矩阵相乘：$$a\odot (1-a)$$，后面不再强调。

推导过程如下：

令：$$u=1，v=1+e^{-z}$$ 则：

$$
\begin{aligned}
Sigmoid^{\prime}(z)=\frac{u^{\prime} v-v^{\prime} u}{v^{2}} &=\frac{0-\left(1+e^{-z}\right)^{\prime}}{\left(1+e^{-z}\right)^{2}} \\
=& \frac{e^{-z}}{\left(1+e^{-z}\right)^{2}}=\frac{1+e^{-z}-1}{\left(1+e^{-z}\right)^{2}} \\
=& \frac{1}{1+e^{-z}}-\left(\frac{1}{1+e^{-z}}\right)^{2} \\
=& a-a^{2}=a(1-a)
\end{aligned}
$$

### 值域

* 输入值域：$$(-\infty, \infty)$$
* 输出值域：$$(0,1)$$
* 导数值域：$$[0,0.25]$$

### 函数图像

![&#x56FE;8-3 Sigmoid&#x51FD;&#x6570;&#x56FE;&#x50CF;](../.gitbook/assets/image%20%28167%29.png)

### 优点

从函数图像来看，Sigmoid函数的作用是将输入压缩到\(0, 1\)这个区间范围内，这种输出在0~1之间的函数可以用来模拟一些概率分布的情况。他还是一个连续函数，导数简单易求。

从数学上来看，Sigmoid函数对中央区的信号增益较大，对两侧区的信号增益小，在信号的特征空间映射上，有很好的效果。

从神经科学上来看，中央区酷似神经元的兴奋态，两侧区酷似神经元的抑制态，因而在神经网络学习方面，可以将重点特征推向中央区， 将非重点特征推向两侧区。

分类功能：我们经常听到这样的对白：

* 甲：“你觉得这件事情成功概率有多大？”
* 乙：“我有六成把握能成功。”

Sigmoid函数在这里就起到了如何把一个数值转化成一个通俗意义上的“把握”的表示。z坐标值越大，经过Sigmoid函数之后的结果就越接近1，把握就越大。

### 缺点

指数计算代价大。

**反向传播时梯度消失**：从梯度图像中可以看到，Sigmoid的梯度在两端都会接近于0，根据链式法则，如果传回的误差是$$\delta$$，那么梯度传递函数是$$\delta \cdot a'$$，而$$a'$$这时接近零，也就是说整体的梯度也接近零。这就出现梯度消失的问题，**并且这个问题可能导致网络收敛速度比较慢。**

给个纯粹数学的例子，假定我们的学习速率是0.2，Sigmoid函数值是0.9（处于饱和区了），如果我们想把这个函数的值降到0.5，需要经过多少步呢？

我们先来做数值计算：

1. 求出当前输入的值

$$a=\frac{1}{1 + e^{-z}} = 0.9$$

$$z = ln{9}$$

1. 求出当前梯度

$$\delta = a \times (1 - a) = 0.9 \times 0.1= 0.09$$

1. 根据梯度更新当前输入值

$$z_{new} = z - \eta \times \delta = ln{9} - 0.2 \times 0.09 = ln(9) - 0.018$$

1. 判断当前函数值是否接近0.5

$$a=\frac{1}{1 + e^{-z_{new}}} = 0.898368$$

1. 重复步骤2-3，直到当前函数值接近0.5

如果用一个程序来计算的话，需要迭代67次，才可以从0.9趋近0.5。如果对67次这个数字没概念的话，读者可以参看接下来关于[ReLU函数](ban-xian-xing-ji-huo-han-shu.md#relu-han-shu)的相关介绍。

此外，**如果输入数据是\(-1, 1\)范围内的均匀分布的数据会导致什么样的结果呢？经过Sigmoid函数处理之后这些数据的均值就从0变到了0.5，导致了均值的漂移**，在很多应用中，这个性质是不好的。

## Tanh函数

TanHyperbolic，即双曲正切函数。

### 公式

$$Tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}} = (\frac{2}{1 + e^{-2z}}-1) \rightarrow a \tag{3}$$

 即 

$$Tanh(z) = 2 \cdot Sigmoid(2z) - 1 \tag{4}$$

### 导数公式

$$
Tanh'(z) = (1 + a)(1 - a)
$$

利用基本导数公式23，令：$$u={e^{z}-e^{-z}}，v=e^{z}+e^{-z}$$ 则有：

$$
\begin{array}{c}
Tanh^{\prime}(z)=\frac{u^{\prime} v-v^{\prime} u}{v^{2}} \\
=\frac{\left(e^{z}-e^{-z}\right)^{\prime}\left(e^{z}+e^{-z}\right)-\left(e^{z}+e^{-z}\right)^{\prime}\left(e^{z}-e^{-z}\right)}{\left(e^{z}+e^{-z}\right)^{2}} \\
=\frac{\left(e^{z}+e^{-z}\right)\left(e^{z}+e^{-z}\right)-\left(e^{z}-e^{-z}\right)\left(e^{z}-e^{-z}\right)}{\left(e^{z}+e^{-z}\right)^{2}} \\
=\frac{\left(e^{z}+e^{-z}\right)^{2}-\left(e^{z}-e^{-z}\right)^{2}}{\left(e^{z}+e^{-z}\right)^{2}} \\
=1-\left(\frac{\left(e^{z}-e^{-z}\right.}{e^{z}+e^{-z}}\right)^{2}=1-a^{2}
\end{array}
$$

### 值域

* 输入值域：$$(-\infty, \infty)$$
* 输出值域：$$(-1,1)$$
* 导数值域：$$[0,1]$$

### 函数图像

图8-4是双曲正切的函数图像。

![&#x56FE;8-4 &#x53CC;&#x66F2;&#x6B63;&#x5207;&#x51FD;&#x6570;&#x56FE;&#x50CF;](../.gitbook/assets/image%20%28171%29.png)

### 优点

具有Sigmoid的所有优点。

无论从理论公式还是函数图像，这个函数都是一个和Sigmoid非常相像的激活函数，他们的性质也确实如此。但是比起sigmoid，tanh减少了一个缺点，就是他本身是零均值的，也就是说，在传递过程中，输入数据的均值并不会发生改变，这就使他在很多应用中能表现出比Sigmoid优异一些的效果。

### 缺点

**exp指数计算代价大。梯度消失问题仍然存在。**

## 其它函数

图8-5展示了其它S型函数，除了tanh\(x\)以外，其它的基本不怎么使用，目的是告诉大家这类函数有很多，但是常用的只有Sigmoid和Tanh两个。

![&#x56FE;8-5 &#x5176;&#x5B83;S&#x578B;&#x51FD;&#x6570;](../.gitbook/assets/image%20%28172%29.png)

再强调一下本书中的约定：

1. Sigmoid，指的是对数几率函数用于激活函数时的称呼；
2. Logistic，指的是对数几率函数用于二分类函数时的称呼；
3. Tanh，指的是双曲正切函数用于激活函数时的称呼。

## 代码位置

原代码位置：[ch08, Level1](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch08-ActivationFunction/Level1_DrawActivators1.py)

个人代码：[**DrawActivators1**](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/ActivationFunction/DrawActivators1.py)\*\*\*\*

