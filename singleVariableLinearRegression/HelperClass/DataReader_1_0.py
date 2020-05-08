# -*- coding: utf-8 -*-#
'''
# Name:         DataReader_1_0
# Description:  读取文件
# Author:       super
# Date:         2020/5/8
'''

import numpy as np
from pathlib import Path

class DataReader_1_0(object):
    def __init__(self, data_file):
        self.train_file_name = data_file
        self.num_train = 0
        self.XTrain = None
        self.YTrain = None

    # read data from file
    def ReadData(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = np.load(self.train_file_name)
            # 读取训练数据，特征与标签
            self.XTrain = data["data"]
            self.YTrain = data["label"]
            self.num_train = self.XTrain.shape[0]
        else:
            raise Exception("Cannot find train file!!!")
        #end if

    # get single training data
    # 获取单个数据
    def GetSingleTrainSample(self, iteration):
        x = self.XTrain[iteration]
        y = self.YTrain[iteration]
        return x, y

    # get batch training data
    # 批量获取数据
    def GetBatchTrainSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_X = self.XTrain[start:end,:]
        batch_Y = self.YTrain[start:end,:]
        return batch_X, batch_Y

    #获取整个数据集
    def GetWholeTrainSamples(self):
        return self.XTrain, self.YTrain

    # permutation only affect along the first axis,
    # so we need transpose the array first
    #
    # >>> arr = np.arange(9).reshape((3, 3))
    # >>> np.random.permutation(arr)
    # array([[6, 7, 8],
    #        [0, 1, 2],
    #        [3, 4, 5]])
    # only affect the first axis
    # see the comment of this class to understand the data format
    # 打乱数据内容
    def Shuffle(self):
        seed = np.random.randint(0,100)
        np.random.seed(seed)
        XP = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        YP = np.random.permutation(self.YTrain)
        self.XTrain = XP
        self.YTrain = YP