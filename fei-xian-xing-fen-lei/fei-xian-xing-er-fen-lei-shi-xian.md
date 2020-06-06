# 非线性二分类实现

## 定义神经网络结构

首先定义可以完成非线性二分类的神经网络结构图，如图10-6所示。

![&#x56FE;10-6 &#x975E;&#x7EBF;&#x6027;&#x4E8C;&#x5206;&#x7C7B;&#x795E;&#x7ECF;&#x7F51;&#x7EDC;&#x7ED3;&#x6784;&#x56FE;](../.gitbook/assets/image%20%28198%29.png)

* 输入层两个特征值x1, x2 

$$
X=\begin{pmatrix} x_1 & x_2 \end{pmatrix}
$$

* 隐层2x2的权重矩阵W1 

$$
W1=\begin{pmatrix}
    w^1_{11} & w^1_{12} \\
    w^1_{21} & w^1_{22} 
  \end{pmatrix}
$$

* 隐层1x2的偏移矩阵B1

$$
B1=\begin{pmatrix}
    b^1_{1} & b^1_{2}
  \end{pmatrix}
$$

* 隐层由两个神经元构成 

$$
Z1=\begin{pmatrix} z1_{1} & z1_{2} \end{pmatrix}
$$

$$
A1=\begin{pmatrix} a1_{1} & a1_{2} \end{pmatrix}
$$

* 输出层2x1的权重矩阵W2 

$$
W2=\begin{pmatrix}
    w^2_{11} \\
    w^2_{21}  
  \end{pmatrix}
$$

* 输出层1x1的偏移矩阵B2

$$
B2=\begin{pmatrix}
    b^2_{1}
  \end{pmatrix}
$$

* 输出层有一个神经元使用Logisitc函数进行分类 

$$
Z2=\begin{pmatrix}
    z^2_{1}
  \end{pmatrix}
$$

$$
A2=\begin{pmatrix}
    a^2_{1}
  \end{pmatrix}
$$

对于一般的用于二分类的双层神经网络可以是图10-7的样子。

![&#x56FE;10-7 &#x901A;&#x7528;&#x7684;&#x4E8C;&#x5206;&#x7C7B;&#x795E;&#x7ECF;&#x7F51;&#x7EDC;&#x7ED3;&#x6784;&#x56FE;](../.gitbook/assets/image%20%28234%29.png)

输入特征值可以有很多，隐层单元也可以有很多，输出单元只有一个，且后面要接Logistic分类函数和二分类交叉熵损失函数。

## 前向计算

根据网络结构，我们有了前向计算过程图10-8。

![&#x56FE;10-8 &#x524D;&#x5411;&#x8BA1;&#x7B97;&#x8FC7;&#x7A0B;](../.gitbook/assets/image%20%28202%29.png)

### 第一层

* 线性计算

$$
z^1_{1} = x_{1} w^1_{11} + x_{2} w^1_{21} + b^1_{1}
$$

$$
z^1_{2} = x_{1} w^1_{12} + x_{2} w^1_{22} + b^1_{2}
$$

$$
Z1 = X \cdot W1 + B1
$$

* 激活函数

$$
a^1_{1} = Sigmoid(z^1_{1})
$$

$$
a^1_{2} = Sigmoid(z^1_{2})
$$

$$
A1=\begin{pmatrix}
  a^1_{1} & a^1_{2}
\end{pmatrix}
$$

### 第二层

* 线性计算

$$
z^2_1 = a^1_{1} w^2_{11} + a^1_{2} w^2_{21} + b^2_{1}
$$

$$
Z2 = A1 \cdot W2 + B2
$$

* 分类函数

$$
a^2_1 = Logistic(z^2_1)
$$

$$
A2 = Logistic(Z2)
$$

### 损失函数

我们把异或问题归类成二分类问题，所以使用二分类交叉熵损失函数：

$$ loss = -Y \ln A2 + (1-Y) \ln (1-A2) \tag{12} $$

在二分类问题中，Y、A2都是一个单一的数值，而非矩阵，但是为了前后统一，我们可以把它们看作是一个1x1的矩阵。

## 反向传播

图10-9展示了反向传播的过程。

![&#x56FE;10-9 &#x53CD;&#x5411;&#x4F20;&#x64AD;&#x8FC7;&#x7A0B;](../.gitbook/assets/image%20%28199%29.png)

### 求损失函数对输出层的反向误差

对损失函数求导，可以得到损失函数对输出层的梯度值，即图10-9中的Z2部分。

根据公式12，求A2和Z2的导数（此处A2、Z2、Y可以看作是标量，以方便求导）：

$$ \begin{aligned} {\partial loss \over \partial Z2}={\partial loss \over \partial A2}{\partial A2 \over \partial Z2} \ ={A2-Y \over A2(1-A2)} \cdot A2(1-A2) \ =A2-Y \rightarrow dZ2 \end{aligned} \tag{13} $$

### 求W2和B2的梯度

$$
{\partial loss \over \partial W2}=\begin{pmatrix}  {\partial loss \over \partial w^2_{11}} \\  \\  {\partial loss \over \partial w^2_{21}}\end{pmatrix}=\begin{pmatrix}  {\partial loss \over \partial Z2}{\partial z2 \over \partial w^2_{11}} \\  \\  {\partial loss \over \partial Z2}{\partial z2 \over \partial w^2_{21}}\end{pmatrix}
$$

$$
=\begin{pmatrix}
  dZ2 \cdot a^1_{1} \\
  dZ2 \cdot a^1_{2} 
\end{pmatrix}
=\begin{pmatrix}
  a^1_{1} \\ a^1_{2}
\end{pmatrix}dZ2
$$

$$
=A1^T \cdot dZ2 => dW2  \tag{14}
$$

$$
{\partial{loss} \over \partial{B2}}=dZ2 => dB2 \tag{15}
$$

### 求损失函数对隐层的反向误差

$$
\frac{\partial{loss}}{\partial{A1}} = \begin{pmatrix}  {\partial loss \over \partial a^1_{1}} & {\partial loss \over \partial a^1_{2}} \end{pmatrix}
$$

$$
=\begin{pmatrix}
\frac{\partial{loss}}{\partial{Z2}} \frac{\partial{Z2}}{\partial{a^1_{1}}} & \frac{\partial{loss}}{\partial{Z2}}  \frac{\partial{Z2}}{\partial{a^1_{2}}}  
\end{pmatrix}
$$

$$
=\begin{pmatrix}
dZ2 \cdot w^2_{11} & dZ2 \cdot w^2_{21}
\end{pmatrix}
$$

$$
=dZ2 \cdot \begin{pmatrix}
  w^2_{11} & w^2_{21}
\end{pmatrix}
$$

$$
=dZ2 \cdot W2^T \tag{16}
$$

$$
{\partial A1 \over \partial Z1}=A1 \odot (1-A1) => dA1\tag{17}
$$

所以最后到达z1的误差矩阵是：

$$ \begin{aligned} {\partial loss \over \partial Z1}={\partial loss \over \partial A1}{\partial A1 \over \partial Z1} \ =dZ2 \cdot W2^T \odot dA1 \rightarrow dZ1 \end{aligned} \tag{18} $$

有了dZ1后，再向前求W1和B1的误差，我们直接列在下面了：

$$
dW1=X^T \cdot dZ1 \tag{19}
$$

$$
dB1=dZ1 \tag{20}
$$

