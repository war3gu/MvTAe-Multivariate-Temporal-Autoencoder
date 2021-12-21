

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

class window:
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

        self.norm_x = None
        self.norm_y = None

    def norm(self):
        self.norm_x = pd.DataFrame()
        for j, feature in enumerate(features_x):
            ret_x, _hi, _lo = norm(self.data_x[feature][:])
            self.norm_x[feature] = ret_x
        _, hi, lo = norm(self.data_x[feature_y][:])                #此处使用norm函数返回的最大最小
        self.norm_y = norm(self.data_y[feature_y], hi, lo)[0]                #xxx是tuple（list列表和tuple元组的“技术差异”是，list列表是可变的，而tuple元组是不可变的。这是在 Python 语言中二者唯一的差别。(所以tuple大多数情况比list快)）

    def get_norm_data_frame(self):
        return self.norm_x, self.norm_y

    def get_norm_data_array(self):
        return np.array(self.norm_x), self.norm_y

    def get_raw_data(self, norm_pre_y):                           #还原数据
        _, hi, lo = norm(self.data_x[feature_y][:])
        raw_pre_y = reverse_norm(norm_pre_y, hi, lo)
        return self.data_y[feature_y], raw_pre_y