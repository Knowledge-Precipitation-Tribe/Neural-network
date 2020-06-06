# 多入多出的三层神经网络

## 多变量非线性多分类

### 提出问题

手写识别是人工智能的重要课题之一。MNIST数字手写体识别图片集，大家一定不陌生，下面就是一些样本。

![&#x56FE;12-1 MNIST&#x6570;&#x636E;&#x96C6;&#x6837;&#x672C;&#x793A;&#x4F8B;](../.gitbook/assets/image%20%28279%29.png)

由于这是从欧美国家和地区收集的数据，从图中可以看出有几点和中国人的手写习惯不一样：

* 数字2，下面多一个圈
* 数字4，很多横线不出头
* 数字6，上面是直的
* 数字7，中间有个横杠

不过这些细节不影响咱们学习课程，正好还可以验证一下中国人的手写习惯是否能够被正确识别。

由于不是让我们识别26个英文字母或者3500多个常用汉字，所以问题还算是比较简单，不需要图像处理知识，也暂时不需要卷积神经网络的参与。咱们可以试试用一个三层的神经网络解决此问题，把每个图片的像素都当作一个向量来看，而不是作为点阵。

#### 图片数据归一化

数据归一化，是针对数据的特征值做的处理，也就是针对样本数据的列做的处理。

本章中，要处理的对象是图片，需要把整张图片看作一个样本，因此使用下面这段代码做数据归一化方法：

```python
    def __NormalizeData(self, XRawData):
        X_NEW = np.zeros(XRawData.shape)
        x_max = np.max(XRawData)
        x_min = np.min(XRawData)
        X_NEW = (XRawData - x_min)/(x_max-x_min)
        return X_NEW
```

### 代码位置

原代码位置：[ch12, MnistDataReader.py](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch12-MultipleLayerNetwork/HelperClass2/MnistImageDataReader.py)

个人代码：[**MnistImageDataReader**](https://github.com/Knowledge-Precipitation-Tribe/Neural-network/blob/master/MultipleLayerNetwork/HelperClass2/MnistImageDataReader.py)\*\*\*\*

