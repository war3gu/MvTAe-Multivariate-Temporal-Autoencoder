


import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy


from window import window

from defines import *

from mvtae_model import MVTAEModel

import matplotlib.pyplot as plt

print(torch.__version__)
print(torch.cuda.is_available())



#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[2]:


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


# # Load Data

# In[3]:


data = pd.read_csv('data.csv', header=0)
data


# In[4]:
#from pylab import *

#figsize(15,6)
idx_val_split = int(0.8 * data.shape[0])
plt.plot(data['combined'], label='combined signal')
plt.plot(data['sine_1'], label='sine_1 signal')
plt.plot(data['sine_2'], label='sine_2 signal')
plt.plot(data['noise'], alpha=0.5, label='noise signal')
plt.vlines(idx_val_split, ymin=np.min(data['combined']), ymax=np.max(data['combined']), color='r', label='train/val split')
plt.legend()
plt.show()


# # Transform & Normalize Data

# In[5]:

#显示第一个窗口的数据

#figsize(15,6)


plt.plot(norm(data['sine_1'][:window_size])[0], label='sine_1')
plt.plot(norm(data['sine_2'][:window_size])[0], label='sine_2')
plt.plot(norm(data['combined'][:window_size])[0], label='combined')
plt.legend()
plt.show()


# In[6]:
idx_front = 0
idx_rear = window_size



raw_data_size = data.shape[0]
index_window_start = 0
index_window_end = index_window_start + window_size

list_window = []

list_x = []
list_y = []



while index_window_end < raw_data_size:
    data_x_temp = data[index_window_start:index_window_end]
    data_y_temp = data.iloc[index_window_end]

    oneWin = window(data_x_temp, data_y_temp)

    oneWin.norm()

    ret_x_df, ret_y_df = oneWin.get_norm_data_frame()

    ret_x_arr, ret_y_arr = oneWin.get_norm_data_array()

    list_x.append(ret_x_arr)

    list_y.append(ret_y_arr)

    list_window.append(oneWin)

    index_window_start = index_window_start + step_size
    index_window_end = index_window_start + window_size

count_window = len(list_x)
count_train = int(np.ceil(count_window * split_ratio))
count_test = count_window - count_train

#wwwwww = np.array(list_x)
#xxxxxx = wwwwww[:count_train]                 #array也是可以这样分割的


train_arr_x = np.array(list_x[:count_train])
train_arr_y = np.array(list_y[:count_train])

test_arr_x = np.array(list_x[count_train:])
test_arr_y = np.array(list_y[count_train:])


tr_input_seq = train_arr_x
tr_data_windows_y = train_arr_y
#tr_data_size = tr_input_seq.shape[0]                   #3899




plt.subplot(2,2,1)
for i in range(30):
    plt.plot(tr_input_seq[i,:,0])
plt.title('Data Windows sine_1')

plt.subplot(2,2,2)
for i in range(30):
    plt.plot(tr_input_seq[i,:,1])
plt.title('Data Windows sine_2')

plt.subplot(2,1,2)
# test prediction Y with visual
p = [None for x in range(9)]
xxxx = data[feature_y][:window_size]             #取得第一个窗口的所有combined
dddd = norm(xxxx)[0]                              #归一化所有combined
p.append(dddd.iloc[-1])                           #最后一个归一化的combined加入p（data中索引是window_size-1）
p.append(tr_data_windows_y[0])                    #归一化的y加入p（data中索引是window_size）
gggg = list(dddd[-10:])                           #取第一个窗口的最后10个combined（combined和y表示同一个变量）
plt.plot(gggg, label='X')
plt.plot(p, label='Y')
plt.legend()
plt.title('X, Y Window')
plt.show()


data_x = from_numpy(tr_input_seq).float()              #``self.float()`` is equivalent to ``self.to(torch.float32)``
data_y = from_numpy(tr_data_windows_y).float()

dataset = Data(
    x=data_x,
    y=data_y
)

data_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size
)

model = MVTAEModel(model_save_path='./',
                   seq_len=tr_input_seq.shape[1],
                   in_data_dims=tr_input_seq.shape[2],
                   out_data_dims=tr_input_seq.shape[2],
                   model_name='mvtae_model',
                   hidden_vector_size=hidden_vector_size,
                   hidden_alpha_size=hidden_alpha_size,
                   dropout_p=0.1,
                   optim_lr=0.0001)

print('-'*30)
print('Data Batch Size:\t%.2fMB' % calc_input_memory((batch_size, tr_input_seq.shape[1], tr_input_seq.shape[2])))
print('Model Size:\t\t%.2fMB' % calc_model_memory(model))
print('Model Parameters:\t%d' % calc_model_params(model))
print('-'*30)
print('Data Size:\t\t', len(dataset))
print('Batches per Epoch:\t', int(len(dataset)/tr_input_seq.shape[0]))




tensorboard_folder = './tensorboard'

model.fit(data_loader, epochs=3000, start_epoch=0, verbose=True)

model.load_state_dict(torch.load('mvtae_model_best.pth'))


# # Hidden State Vector Visualisation           #显示部分输入数据的hidden state vector

def hidden_state_verctor_visual():
    model.eval()
    zoom_lim = 10 # limit num of hidden vector examples to show
    hidden_state_vector, decoder_output, alpha_output = model(from_numpy(tr_input_seq[:zoom_lim]).float())
    # visualise hidden vector
    for i in range(tr_input_seq[:zoom_lim].shape[0]):
        plt.plot(hidden_state_vector[0, i].detach().cpu())
    plt.show()

hidden_state_verctor_visual()


# # Decoder Target Recreation                   #重构一个输入，然后与原始输入同时显示对比

def decoder_scores():
    model.eval()
    idx = 5                                         #进行重构的输入的索引
    y_in = tr_input_seq[idx].reshape(1, window_size, tr_input_seq.shape[2])
    hidden_state_vector, decoder_output, alpha_output = model(from_numpy(y_in).float())
    plt.subplot(3,1,1)
    plt.plot(tr_input_seq[idx,:,0], label='Target')
    plt.plot(decoder_output[0,:,0].detach().cpu().numpy()[::-1], label='Prediction')
    plt.title('sine_1')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(tr_input_seq[idx,:,1], label='Target')
    plt.plot(decoder_output[0,:,1].detach().cpu().numpy()[::-1], label='Prediction')
    plt.title('sine_2')
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(tr_input_seq[idx,:,2], label='Target')
    plt.plot(decoder_output[0,:,2].detach().cpu().numpy()[::-1], label='Prediction')
    plt.title('noise')
    plt.legend()
    plt.show()

decoder_scores()


# # Alpha Target Branch Prediction                 #使用tr_input_seq进行预测，与结果tr_data_windows_y进行比较。此处没有使用test数据，感觉不太好.

def alpha_train_scores():
    model.eval()
    _,_ , alpha_output = model(from_numpy(tr_input_seq).float())
    alpha_output = alpha_output.flatten().detach().cpu().numpy()
    print('###  Train  Error/Accuracy Metrics ###')
    print('MSE:\t', mean_squared_error(tr_data_windows_y, alpha_output))
    print('MAE:\t', mean_absolute_error(tr_data_windows_y, alpha_output))
    print('R²:\t', r2_score(tr_data_windows_y, alpha_output))
    plt.plot(tr_data_windows_y, label='Target')
    plt.plot(alpha_output, label='Prediction')
    plt.legend()
    plt.show()

alpha_train_scores()


##进行纯正的test，看看具体的情况
def alpha_test_scores():
    model.eval()
    _,_ , alpha_output = model(from_numpy(test_arr_x).float())
    alpha_output = alpha_output.flatten().detach().cpu().numpy()
    print('###  Test  Error/Accuracy Metrics ###')
    print('MSE:\t', mean_squared_error(test_arr_y, alpha_output))
    print('MAE:\t', mean_absolute_error(test_arr_y, alpha_output))
    print('R²:\t', r2_score(test_arr_y, alpha_output))
    plt.plot(test_arr_y, label='Target')
    plt.plot(alpha_output, label='Prediction')
    plt.legend()
    plt.show()
alpha_test_scores()

k = 0
k += 1
