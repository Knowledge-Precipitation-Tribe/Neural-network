# 梯度下降法

有了上一节的最小二乘法做基准，我们这次用梯度下降法求解w和b，从而可以比较二者的结果。

## 数学原理

在下面的公式中，我们规定x是样本特征值（单特征），y是样本标签值，z是预测值，下标 $$i$$ 表示其中一个样本。

### 预设函数（Hypothesis Function）

为一个线性函数：

$$z_i = x_i \cdot w + b \tag{1}$$

### 损失函数（Loss Function）

为均方差函数：

$$loss(w,b) = \frac{1}{2} (z_i-y_i)^2 \tag{2}$$

与最小二乘法比较可以看到，**梯度下降法和最小二乘法的模型及损失函数是相同的**，都是一个线性模型加均方差损失函数，模型用于拟合，损失函数用于评估效果。

区别在于，**最小二乘法从损失函数求导，直接求得数学解析解**，而**梯度下降以及后面的神经网络，都是利用导数传递误差，再通过迭代方式一步一步（用近似解）逼近真实解**。

## 梯度计算

### 计算z的梯度

根据公式2： $$ {\partial loss \over \partial z_i}=z_i - y_i \tag{3} $$

### 计算w的梯度

我们用loss的值作为误差衡量标准，通过求w对它的影响，也就是loss对w的偏导数，来得到w的梯度。由于loss是通过公式2-&gt;公式1间接地联系到w的，所以我们使用链式求导法则，通过单个样本来求导。

根据公式1和公式3：

$$ {\partial{loss} \over \partial{w}} = \frac{\partial{loss}}{\partial{z_i}}\frac{\partial{z_i}}{\partial{w}}=(z_i-y_i)x_i \tag{4} $$

### 计算b的梯度

$$ \frac{\partial{loss}}{\partial{b}} = \frac{\partial{loss}}{\partial{z_i}}\frac{\partial{z_i}}{\partial{b}}=z_i-y_i \tag{5} $$

## 代码实现

```python
if __name__ == '__main__':

    reader = SimpleDataReader()
    reader.ReadData()
    X,Y = reader.GetWholeTrainSamples()

    eta = 0.1
    w, b = 0.0, 0.0
    for i in range(reader.num_train):
        # get x and y value for one sample
        xi = X[i]
        yi = Y[i]
        # 公式1
        zi = xi * w + b
        # 公式3
        dz = zi - yi
        # 公式4
        dw = dz * xi
        # 公式5
        db = dz
        # update w,b
        w = w - eta * dw
        b = b - eta * db

    print("w=", w)    
    print("b=", b)
```

大家可以看到，在代码中，我们完全按照公式推导实现了代码，所以，大名鼎鼎的梯度下降，其实就是**把推导的结果转化为数学公式和代码，直接放在迭代过程里**！另外，我们**并没有直接计算损失函数值，而只是把它融入在公式推导中。**

## 运行结果

```text
w= [1.71629006]
b= [3.19684087]
```

读者可能会注意到，上面的结果和最小二乘法的结果（w1=2.056827, b1=2.965434）相差比较多，这个问题我们留在本章稍后的地方解决。

## 代码位置

原代码位置：[ch04, Level2](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch04-SingleVariableLinearRegression/Level2_GradientDescent.py)

个人代码：[GradientDescent](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/singleVariableLinearRegression/LeastSquare.py)

