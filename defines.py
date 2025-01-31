
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy

import random
from defines_ta_id import *

ID_SEED = 42

random.seed(ID_SEED)
np.random.seed(ID_SEED)
torch.manual_seed(ID_SEED)
torch.cuda.manual_seed(ID_SEED)
torch.cuda.manual_seed_all(ID_SEED)


type_model = "binary"



time_model = "min1"    #min5,min1,day

features_x = ['sine_1', 'sine_2', 'noise']
feature_y = 'combined'

step_size = 1                 #窗口移动的步长
float_precision_bits = 32
bits_in_MB = 8e6







id_market   = 'SSE'

class macro():
    window_size = 100             #窗口的大小               可配置
    weight_decoder = 10           #                       可配置
    weight_alpha   = 10           #                       可配置
    hidden_vector_size = 64       #                       可配置
    hidden_alpha_size = 16        #                       可配置
    batch_size = 1024             #                       可配置
    epochs_size = 10              #                       可配置
    dropout_p   = 0.1             #                       可配置
    lr          = 0.0001          #                       可配置
    weight_decay= 0.8             #                       可配置
    id_stock    = '600196.sh'     #                       可配置
    split_ratio = 0.9             #                       可配置
    data_size   = 0               #使用的窗口数据的大小       可配置


index_list_super_params = [3]

#39,7
index_list_super_ta_params =[TAID_NONE]


FIELD_DATE = 'trade_date'

features_stock_0 = ['open','high','low','close','vol','amount','pre_close','change','pct_chg']

features_stock_1 = ['open','high','low','close','vol']

features_day = ['open','high','low','close','vol','amount']

class macroFeature:
    features_x_norm = ['open','high','low','close', 'amount', 'volume']
    features_x = ['open', 'high', 'low', 'close', 'amount', 'volume']
              #'CUMLOGRET_1',  'CUMPCTRET_1', 'EMA_60', 'SMA_60',
              #'MACD_12_26_9', 'MACDh_12_26_9', 'RSI_60',
              #'K_60_3', 'D_60_3', 'J_60_3']
    feature_y = 'close'

#features_stock = features_stock_1

#features_size = len(features_stock)                       #属性池长度

#stock_arr_yiyao = ['600196.sh','600276.sh','002821.sz','002001.sz']

#stock_arr_yiyao1 = ['600196.sh','000963.sz','600276.sh','600079.sh','002422.sz','600380.sh','600420.sh','002001.sz',
                    #'600664.sh','000513.sz','600267.sh','600673.sh','600062.sh','600216.sh','000739.sz','002793.sz']

#stock_arr_jiadian = ['000333.sz', '600690.sh', '000651.sz']

#stock_arr_0 = ['000333.sz', '600196.sh', '600519.sh', '600104.sh']



stocks_list_old = ['000507.sz',
               '000006.sz',
               '000005.sz',
               '000011.sz',
               '000016.sz',
               '000020.sz',
               '000021.sz',
               '000029.sz',
               '000036.sz',
               '000032.sz',
               '000407.sz',
               '000506.sz',
               '000524.sz',
               '000534.sz',
               '000548.sz',
               '000554.sz',
               '000571.sz',
               '000573.sz',
               '000589.sz',
               '000610.sz']

stocks_list =['600104.sh',
              '600188.sh',
              '600196.sh',
              '600276.sh',
              '600436.sh',
              '600507.sh',
              '600519.sh',
              '600660.sh',
              '600771.sh',
              '601318.sh',
              '000001.sz',
              '000063.sz',
              '000333.sz',
              '000651.sz',
              '000858.sz',
              '000963.sz',
              '002415.sz',
              '002714.sz',
              '300142.sz',
              '600009.sh']

data_folder = './data'

token_TS = '51295be6098fe565f6f727019e280ba4821ad5554b551c311bc33ae3'
