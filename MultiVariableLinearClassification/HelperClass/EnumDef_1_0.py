# -*- coding: utf-8 -*-#
'''
# Name:         EnumDef
# Description:  判断网络类型
# Author:       super
# Date:         2020/5/16
'''
from enum import Enum

# 判断网络类型
class NetType(Enum):
    Fitting = 1,
    BinaryClassifier = 2,
    MultipleClassifier = 3
    BinaryTanh = 4