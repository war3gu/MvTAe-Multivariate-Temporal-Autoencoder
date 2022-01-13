

import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from defines import *
from utils   import *

class window:
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

        self.norm_x = None
        self.norm_y = None

    def norm(self):
        self.norm_x = pd.DataFrame()
        for j, feature in enumerate(features_x):
            if feature in features_x_norm:
                self.norm_x[feature] = norm_x(self.data_x[feature][:])
            else:
                self.norm_x[feature] = self.data_x[feature][:]
        hi, lo = get_hi_lo(self.data_x[feature_y][:])                #此处使用norm函数返回的最大最小
        self.norm_y = norm_y(self.data_y[feature_y], hi, lo)                #xxx是tuple（list列表和tuple元组的“技术差异”是，list列表是可变的，而tuple元组是不可变的。这是在 Python 语言中二者唯一的差别。(所以tuple大多数情况比list快)）

    def get_norm_data_frame(self):
        return self.norm_x, self.norm_y

    def get_norm_data_array(self):
        return np.array(self.norm_x), self.norm_y

    def get_up_down_array(self):
        xxx = self.norm_x.iloc[-1]
        vvv = xxx[feature_y]
        if self.norm_y > vvv:
            return np.array(self.norm_x), 1
        else:
            return np.array(self.norm_x), 0

    def get_raw_data(self, norm_pre_y):                           #还原数据
        hi, lo = get_hi_lo(self.data_x[feature_y][:])
        raw_pre_y = reverse_norm(norm_pre_y, hi, lo)
        y_raw_1   = self.data_x[feature_y].iloc[-1]
        #ffff      = self.data_x[feature_y]
        #xxxx      = self.data_x[-1]
        #y_raw_1   = xxxx[feature_y]
        #y_raw_1   = ffff[-1]                    #前一天的收盘价
        return self.data_y[feature_y], raw_pre_y, y_raw_1