# 回归常用评价指标

## 定义

 预测值：

$$
\hat{\mathbf{y}}=\left\{\hat{y}_{1}, \hat{y}_{2}, \ldots, \hat{y}_{n}\right\}
$$

```python
y_pred = np.array([1.0, 3.3, 2.0, 7.9, 5.5, 6.4, 2.0])
```

 真实值：

$$
\mathbf{y}=\left\{y_{1}, y_{2}, \dots, y_{n}\right\}
$$

```python
y_true = np.array([2.0, 3.0, 2.5, 1.0, 4.0, 3.2, 3.0])
```

```python
import numpy as np
from sklearn import metrics
```

## MSE

均方误差（Mean Square Error）

$$
M S E=\frac{1}{n} \sum_{i=1}^{n}\left(\hat{y}_{i}-y_{i}\right)^{2}
$$

```python
metrics.mean_squared_error(y_true, y_pred)

def MSE(y_true, y_pred):
    return np.average((y_true - y_pred) ** 2, axis=0)
```

## RMSE

均方根误差（Root Mean Square Error），与上面的均方误差相比这里只是加了一个跟号。

$$
R M S E=\sqrt{\frac{1}{n} \sum_{i=1}^{n}\left(\hat{y}_{i}-y_{i}\right)^{2}}
$$

```python
np.sqrt(metrics.mean_squared_error(y_true, y_pred))

def RMSE(y_true, y_pred):
    return np.sqrt(np.average((y_true - y_pred) ** 2, axis=0))
```

## MAE

平均绝对误差（Mean Absolute Error）

$$
M A E=\frac{1}{n} \sum_{i=1}^{n}\left|\hat{y}_{i}-y_{i}\right|
$$

```python
metrics.mean_absolute_error(y_true, y_pred)

def MAE(y_true, y_pred):
    return np.average(np.abs(y_pred - y_true), axis=0)
```

## MAPE

平均绝对百分比误差（Mean Absolute Percentage Error）

$$
M A P E=\frac{100 \%}{n} \sum_{i=1}^{n}\left|\frac{\hat{y}_{i}-y_{i}}{y_{i}}\right|
$$

```python
def MAPE(y_true, y_pred):
    return np.average(np.abs((y_pred - y_true) / y_true), axis=0) * 100
```

## SMAPE

对称平均绝对百分比误差（Symmetric Mean Absolute Percentage Error）

$$
S M A P E=\frac{100 \%}{n} \sum_{i=1}^{n} \frac{\left|\hat{y}_{i}-y_{i}\right|}{\left(\left|\hat{y}_{i}\right|+\left|y_{i}\right|\right) / 2}
$$

```python
def SMAPE(y_true, y_pred):
    return 2.0 * np.average(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)), axis=0) * 100
```

