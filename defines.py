
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy

features_x = ['sine_1', 'sine_2', 'noise']
feature_y = 'combined'

window_size = 100          #窗口的大小
step_size = 1              #窗口移动的步长

split_ratio = 0.8          #窗口分割比例


float_precision_bits = 32
bits_in_MB = 8e6

hidden_vector_size = 64
hidden_alpha_size = 16
batch_size = 1024


def norm(data, hi=None, lo=None):          #hi,lo是外部输入的最大最小值，更高优先级（在Python中，None、空列表[]、空字典{}、空元组()、0等一系列代表空和无的对象会被转换成False）
    hi = np.max(data) if not hi else hi    #hi如果等于False，就从data中找最大值（默认是None，就等于False）。否则执行hi=hi
    lo = np.min(data) if not lo else lo
    if hi-lo == 0:
        return 0, hi, lo
    y = (data-lo)/(hi-lo)
    return y, hi, lo

def reverse_norm(y, hi, lo):
    x = y*(hi-lo)+lo
    return x




def calc_input_memory(input_shape):
    input_bits = np.prod(input_shape, dtype='int64')*float_precision_bits
    return input_bits / bits_in_MB

def calc_model_memory(model):
    mods = list(model.modules())
    sizes = []
    for i in range(1,len(mods)):
        m = mods[i]
        p = list(m.parameters())
        for j in range(len(p)):
            sizes.append(np.array(p[j].size()))

    total_bits = 0
    for i in range(len(sizes)):
        s = sizes[i]
        bits = np.prod(np.array(s), dtype='int64')*float_precision_bits
        total_bits += bits
    return total_bits / bits_in_MB

def calc_model_params(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

class Data(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

    def __len__(self):
        return len(self.x)