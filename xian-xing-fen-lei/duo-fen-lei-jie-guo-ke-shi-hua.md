# 多分类结果可视化

神经网络到底是一对一方式，还是一对多方式呢？从Softmax公式，好像是一对多方式，因为只取一个最大值，那么理想中的一对多方式应该是图7-13所示的样子。

![&#x56FE;7-13 &#x7406;&#x60F3;&#x4E2D;&#x7684;&#x4E00;&#x5BF9;&#x591A;&#x65B9;&#x5F0F;&#x7684;&#x5206;&#x5272;&#x7EBF;](../.gitbook/assets/image%20%2892%29.png)

实际上是什么样子的，我们来看下面的具体分析。

## 显示原始数据图

与二分类时同样的问题，如何直观地理解多分类的结果？三分类要复杂一些，我们先把原始数据显示出来。

```python
# 三分类可视化
def DrawThreeCategoryPoints(X1, X2, Y_onehot, xlabel="x1", ylabel="x2", title=None, show=False, isPredicate=False):
    colors = ['b', 'r', 'g']
    shapes = ['s', 'x', 'o']
    assert(X1.shape[0] == X2.shape[0] == Y_onehot.shape[0])
    count = X1.shape[0]
    for i in range(count):
        j = np.argmax(Y_onehot[i])
        if isPredicate:
            plt.scatter(X1[i], X2[i], color=colors[j], marker='^', s=200, zorder=10)
        else:
            plt.scatter(X1[i], X2[i], color=colors[j], marker=shapes[j], zorder=10)
    #end for
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()
```

会画出图7-1来。

![](../.gitbook/assets/image%20%28137%29.png)

## 显示分类结果分割线图

下面的数据是神经网络训练出的权重和偏移值的结果：

```python
......
epoch=98
98 1385 0.25640040547970516
epoch=99
99 1399 0.2549651316913006
W= [[-1.43299777 -3.57488388  5.00788165]
 [ 4.47527075 -2.88799216 -1.58727859]]
B= [[-1.821679    3.66752583 -1.84584683]]
......
```

其实在讲解多分类原理的时候，我们已经解释了其几何理解，那些公式的推导就可以用于指导我们画出多分类的分割线来。先把几个有用的结论拿过来。

从线性多分类中的公式16，把不等号变成等号，即$$z_1=z_2$$，则代表了那条绿色的分割线，用于分割第一类和第二类的：

$$x_2 = {w_{12} - w_{11} \over w_{21} - w_{22}}x_1 + {b_2 - b_1 \over w_{21} - w_{22}} \tag{1}$$

$$
即：y = W_{12} \cdot x + B_{12}
$$

由于Python数组是从0开始的，所以公式1中的所有下标都减去1，写成代码：

```python
b12 = (net.B[0,1] - net.B[0,0])/(net.W[1,0] - net.W[1,1])
w12 = (net.W[0,1] - net.W[0,0])/(net.W[1,0] - net.W[1,1])
```

从线性多分类中的公式17，把不等号变成等号，即$$z_1=z_3$$，则代表了那条红色的分割线，用于分割第一类和第三类的：

$$x_2 = {w_{13} - w_{11} \over w_{21} - w_{23}} x_1 + {b_3 - b_1 \over w_{21} - w_{23}} \tag{2}$$

$$
即：y = W_{13} \cdot x + B_{13}
$$

写成代码：

```python
b13 = (net.B[0,0] - net.B[0,2])/(net.W[1,2] - net.W[1,0])
w13 = (net.W[0,0] - net.W[0,2])/(net.W[1,2] - net.W[1,0])
```

从线性多分类中的公式24，把不等号变成等号，即$$z_2=z_3$$，则代表了那条蓝色的分割线，用于分割第二类和第三类的：

$$x_2 = {w_{13} - w_{12} \over w_{22} - w_{23}} x_1 + {b_3 - b_2 \over w_{22} - w_{23}} \tag{3}$$

$$
即：y = W_{23} \cdot x + B_{23}
$$

写成代码：

```python
b23 = (net.B[0,2] - net.B[0,1])/(net.W[1,1] - net.W[1,2])
w23 = (net.W[0,2] - net.W[0,1])/(net.W[1,1] - net.W[1,2])
```

改一下主函数，增加对以上两个函数ShowData\(\)和ShowResult\(\)的调用，最后可以看到图7-14所示的分类结果图，注意，这个结果图和我们在7.2中分析的一样，只是蓝线斜率不同。

![&#x56FE;7-14 &#x795E;&#x7ECF;&#x7F51;&#x7EDC;&#x7ED8;&#x51FA;&#x7684;&#x5206;&#x7C7B;&#x7ED3;&#x679C;&#x56FE;](../.gitbook/assets/image%20%2890%29.png)

图7-14中的四个三角形的大点是需要我们预测的四个坐标值，其中三个点的分类都比较明确，只有那个蓝色的点看不清在边界那一侧，可以通过在实际的运行结果图上放大局部来观察。

## 理解神经网络的分类方式

做为实际结果，图7-14与我们猜想的图7-13完全不同：

* 蓝色线是2\|3的边界，不考虑第1类
* 绿色线是1\|2的边界，不考虑第3类
* 红色线是1\|3的边界，不考虑第2类

我们只看蓝色的第1类，当要区分1\|2和1\|3时，神经网络实际是用了两条直线（绿色和红色）同时作为边界。那么它是一对一方式还是一对多方式呢？

图7-14的分割线是我们令$$z_1=z_2, z_2=z_3, z_3=z_1$$三个等式得到的，**但实际上神经网络的工作方式不是这样的，它不会单独比较两类，而是会同时比较三类**，这个从Softmax会同时输出三个概率值就可以理解。比如，当我们想得到第一类的分割线时，需要同时满足两个条件：

$$z_1=z_2，且：z_1=z_3 \tag{4}$$

即，同时，找到第一类和第三类的边界。

**这就意味着公式4其实是一个线性分段函数，而不是两条直线**，即图7-15中红色射线和绿色射线所组成的函数。

![&#x56FE;7-15 &#x5206;&#x6BB5;&#x7EBF;&#x6027;&#x7684;&#x5206;&#x5272;&#x4F5C;&#x7528;](../.gitbook/assets/image%20%28109%29.png)

同理，用于分开红色点和其它两类的分割线是蓝色射线和绿色射线，用于分开绿色点和其它两类的分割线是红色射线和蓝色射线。

**训练一对多分类器时，是把蓝色样本当作一类，把红色和绿色样本混在一起当作另外一类**。训练一对一分类器时，是把绿色样本扔掉，只考虑蓝色样本和红色样本。而我们在此并没有这样做，三类样本是同时参与训练的。所以我们只能说神经网络从结果上看，是一种一对多的方式，至于它的实质，我们在后面的非线性分类时再进一步探讨。

## 代码位置

原代码位置：[ch07, Level2](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch07-LinearMultipleClassification/Level2_ShowMultipleResult.py)

个人代码：[**ShowMultipleResult**](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/MultiVariableLinearClassification/ShowMultipleResult.py)\*\*\*\*

