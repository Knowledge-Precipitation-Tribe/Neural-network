# 正规方程解法

英文名是 Normal Equations。

对于线性回归问题，除了前面提到的最小二乘法可以解决一元线性回归的问题外，也可以解决多元线性回归问题。

对于多元线性回归，可以用正规方程来解决，也就是得到一个数学上的解析解。它可以解决下面这个公式描述的问题：

$$y=a_0+a_1x_1+a_2x_2+\dots+a_kx_k \tag{1}$$

## 简单的推导方法

在做函数拟合（回归）时，我们假设函数H为：

$$h(w,b) = b + x_1 w_1+x_2 w_2+...+x_n w_n \tag{2}$$

令$$b=w_0$$，则：

$$h(w) = w_0 + x_1 \cdot w_1 + x_2 \cdot w_2+...+ x_n \cdot w_n\tag{3}$$

公式3中的x是一个样本的n个特征值，如果我们把m个样本一起计算，将会得到下面这个矩阵：

$$H(w) = X \cdot W \tag{4}$$

公式4中的X和W的矩阵形状如下：

$$
X^{(m \times(n+1))}=\left(\begin{array}{ccccc}
1 & x_{1,1} & x_{1,2} & \dots & x_{1, n} \\
1 & x_{2,1} & x_{2,2} & \dots & x_{2, n} \\
\dots & & \\
1 & x_{m, 1} & x_{m, 2} & \dots & x_{m, n}
\end{array}\right) \tag{5}
$$

$$
W^{(n+1)}=\left(\begin{array}{c}
w_{0} \\
w_{1} \\
\cdots \\
w_{n}
\end{array}\right) \tag{6}
$$

然后我们期望假设函数的输出与真实值一致，则有：

$$H(w) = X \cdot W = Y \tag{7}$$

其中，Y的形状如下：

$$
Y^{(m)}=\left(\begin{array}{l}
y_{1} \\
y_{2} \\
\dots \\
y_{m}
\end{array}\right) \tag{8}
$$

直观上看，W = Y/X，但是这里三个值都是矩阵，而矩阵没有除法，所以需要得到X的逆矩阵，用Y乘以X的逆矩阵即可。但是又会遇到一个问题，只有方阵才有逆矩阵，而X不一定是方阵，所以要先把左侧变成方阵，就可能会有逆矩阵存在了。所以，先把等式两边同时乘以X的转置矩阵，以便得到X的方阵：

$$X^T X W = X^T Y \tag{9}$$

其中，$$X^T$$是X的转置矩阵，$$X^T X$$一定是个方阵，并且假设其存在逆矩阵，把它移到等式右侧来：

$$W = (X^T X)^{-1}{X^T Y} \tag{10}$$

至此可以求出W的正规方程。

## 复杂的推导方法

我们仍然使用均方差损失函数：

$$J(w,b) = \sum (z_i - y_i)^2 \tag{11}$$

把b看作是一个恒等于1的feature，并把$$z=XW$$计算公式带入，并变成矩阵形式：

$$J(w) = \sum (x_i w_i -y_i)^2=(XW - Y)^T \cdot (XW - Y) \tag{12}$$

对w求导，令导数为0，就是W的最小值解：

$$
\begin{array}{c}
\frac{\partial J(w)}{\partial w}=\frac{\partial}{\partial w}\left[(X W-Y)^{T} \cdot(X W-Y)\right] \\
=\frac{\partial}{\partial w}\left[\left(W^{T} X^{T}-Y^{T}\right) \cdot(X W-Y)\right] \\
=\frac{\partial}{\partial w}\left[\left(W^{T}X^{T}XW-W^{T}X^{T} Y-Y^{T} X W+Y^{T} Y\right)\right]
\end{array}
$$

求导后（请参考矩阵/向量求导公式）：

第一项的结果是：$$2X^TXW$$（分母布局，denominator layout）

第二项的结果是：$$X^TY$$（分母布局方式，denominator layout）

第三项的结果是：$$X^TY$$（分子布局方式，numerator layout，需要转置$$Y^TX$$）

第四项的结果是：0

再令导数为0：

$$ J'(w)=2X^TXW - 2X^TY=0 \tag{14} $$

$$ X^TXW = X^TY \tag{15} $$

$$ W=(X^TX)^{-1}X^TY \tag{16} $$

结论和公式10一样。

逆矩阵$$(X^TX)^{-1}$$可能不存在的原因是：

1. 特征值冗余，比如$$x_2=x^2_1$$，即正方形的边长与面积的关系，不能做为两个特征同时存在
2. 特征数量过多，比如特征数n比样本数m还要大

以上两点在我们这个具体的例子中都不存在。

## 代码实现

我们把表5-1的样本数据带入方程内。根据公式\(5\)，我们应该建立如下的X,Y矩阵：

$$
X^{(1000 \times 3)}=\left(\begin{array}{ccc}
1 & 10.06 & 60 \\
1 & 15.47 & 74 \\
1 & 18.66 & 46 \\
1 & 5.20 & 77 \\
\dots
\end{array}\right) \tag{17}
$$

$$
Y^{(1000 \times 1)}=\left(\begin{array}{c}
302.86 \\
393.04 \\
270.67 \\
450.59 \\
\ldots
\end{array}\right) \tag{18}
$$

根据公式\(10\)：

$$W = (X^T X)^{-1}{X^T Y} \tag{10}$$

1. X是1000x3的矩阵，X的转置是3x1000，$$X^TX$$生成\(3x3\)的矩阵
2. $$(X^TX)^{-1}$$也是3x3
3. 再乘以$$X^T$$，即\(3x3\)x\(3x1000\)的矩阵，变成3x1000
4. 再乘以Y，Y是1000x1，所以\(3x1000\)x\(1000x1\)变成3x1，就是W的解，其中包括一个偏移值b和两个权重值w，3个值在一个向量里

```python
if __name__ == '__main__':
    reader = SimpleDataReader()
    reader.ReadData()
    X,Y = reader.GetWholeTrainSamples()
    num_example = X.shape[0]
    one = np.ones((num_example,1))
    x = np.column_stack((one, (X[0:num_example,:])))
    a = np.dot(x.T, x)
    # need to convert to matrix, because np.linalg.inv only works on matrix instead of array
    b = np.asmatrix(a)
    c = np.linalg.inv(b)
    d = np.dot(c, x.T)
    e = np.dot(d, Y)
    #print(e)
    b=e[0,0]
    w1=e[1,0]
    w2=e[2,0]
    print("w1=", w1)
    print("w2=", w2)
    print("b=", b)
    # inference
    z = w1 * 15 + w2 * 93 + b
    print("z=",z)
```

## 运行结果

```python
w1= -2.0184092853092226
w2= 5.055333475112755
b= 46.235258613837644
z= 486.1051325196855
```

我们得到了两个权重值和一个偏移值，然后得到房价预测值z=486万元。

至此，我们得到了解析解。我们可以用这个做为标准答案，去验证我们的神经网络的训练结果。

## 代码位置

原代码位置：[ch05, Level1](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch05-MultiVariableLinearRegression/level1_NormalEquation.py)

个人代码：[**NormalEquation**](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/MultiVariableLinearRegression/NormalEquation.py)\*\*\*\*

