

import numpy as np
import math
from defines import *
import numpy as np
import os


def set_1(value):
    return 1

def norm_x(data):          #hi,lo是外部输入的最大最小值，更高优先级（在Python中，None、空列表[]、空字典{}、空元组()、0等一系列代表空和无的对象会被转换成False）
    hi = np.max(data)
    lo = np.min(data)
    if hi-lo == 0:
        rrr = data.copy()
        rrr = rrr.apply(set_1)
        return rrr
    else:
        vvv = (data-lo)/(hi-lo)
        return vvv

def norm_y(data, hi, lo):
    if hi-lo == 0:
        return data/hi
    else:
        vvv = (data-lo)/(hi-lo)
        return vvv


def get_hi_lo(data):
    hi = np.max(data)
    lo = np.min(data)
    return hi, lo

def reverse_norm(y, hi, lo):
    x = y*(hi-lo)+lo
    return x

def zscore(data, mu=None, sigma=None):
    # z = (x-μ)/σ
    mu = np.mean(data) if not mu else mu
    sigma = np.std(data) if not sigma else sigma
    if sigma == 0:
        return 0, mu, 0
    return (data-mu)/sigma, mu, sigma

def reverse_zscore(z, mu, sigma):
    # x = (zσ)+μ
    x = (z*sigma)+mu
    return x

def fill_zero_last(df, indexName):
    dview = df[indexName].view()
    for idx, val in enumerate(dview):
        if val == 0 or math.isnan(val):
            dview[idx] = dview[idx-1]

def log_data(df, indexName):
    dcopy = df[indexName].copy()
    df[indexName] = np.log(dcopy)



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

if __name__ == '__main__':
    print("haha")
    if not os.path.exists("super_params_scores_sort.csv"):
        df = pd.DataFrame(columns=['id_stock', 'index_sp', 'index_sp_ta', 'mse', 'mae', 'r2', 'dev_per_mean', 'dev_per_std', 'per_pp', 'per_nn', 'per_corr', 'epochs', 'epoch_best', 'epoch_best_loss'])   #还需要记录预测与现实的关系
        df.to_csv("super_params_scores_sort.csv", index = False)

    dataframe = pd.read_csv("super_params_scores.csv")
    dataframe = dataframe.sort_values(by="per_corr", ascending=False)
    dataframe.to_csv("super_params_scores_sort.csv", index = False)