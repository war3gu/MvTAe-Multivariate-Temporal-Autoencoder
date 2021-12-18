#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython import get_ipython

#get_ipython().run_line_magic('load_ext', 'tensorboard')
#get_ipython().run_line_magic('pylab', 'inline')

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy

print(torch.__version__)
print(torch.cuda.is_available())

import matplotlib.pyplot as plt

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


#figsize(15,6)
window_size = 100          #窗口的大小
step_size = 1              #窗口移动的步长

plt.plot(norm(data['sine_1'][:window_size])[0], label='sine_1')
plt.plot(norm(data['sine_2'][:window_size])[0], label='sine_2')
plt.plot(norm(data['combined'][:window_size])[0], label='combined')
plt.legend()
plt.show()


# In[6]:


idx_front = 0
idx_rear = window_size
features_x = ['sine_1', 'sine_2', 'noise']
feature_y = 'combined'

sina_1_data_train = data['sine_1'][:idx_val_split]        #sina_1_data_train是series（series 可以看成是多了元素索引的list）
tr_data_windows_size = int(np.ceil((sina_1_data_train.shape[0]-window_size-1)/step_size))
tr_data_windows = np.empty((tr_data_windows_size, len(features_x), window_size))
tr_data_windows_y = np.zeros(tr_data_windows_size)

i = 0
pbar = tqdm(total=tr_data_windows_size-1, initial=i)
while idx_rear + 1 < sina_1_data_train.shape[0]:
    # create x data windows
    for j, feature in enumerate(features_x):
        _data_window, _hi, _lo = norm(data[feature][idx_front:idx_rear])
        tr_data_windows[i][j] = _data_window              #保存归一化后的x
        
    # create y along same normalized scale
    _, hi, lo = norm(data[feature_y][idx_front:idx_rear]) #此处使用norm函数返回的最大最小
    xxx = norm(data[feature_y][idx_rear], hi, lo)         #xxx是tuple（list列表和tuple元组的“技术差异”是，list列表是可变的，而tuple元组是不可变的。这是在 Python 语言中二者唯一的差别。(所以tuple大多数情况比list快)）
    _y = xxx[0]                                           #归一化后的y
    tr_data_windows_y[i] = _y                             #保存归一化后的y
    
    idx_front = idx_front + step_size
    idx_rear = idx_front + window_size
    i += 1
    pbar.update(1)
pbar.close()

# reshape input into [samples, timesteps, features]
tr_data_size = tr_data_windows.shape[0]
tr_input_seq = tr_data_windows.swapaxes(1,2)              #交换后面2个维度


# In[7]:


#figsize(15,10)
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
xxxx = data['combined'][:window_size]             #取得第一个窗口的所有combined
dddd = norm(xxxx)[0]                              #归一化所有combined
p.append(dddd.iloc[-1])                           #最后一个归一化的combined加入p（data中索引是window_size-1）
p.append(tr_data_windows_y[0])                    #归一化的y加入p（data中索引是window_size）
gggg = list(dddd[-10:])                           #取第一个窗口的最后10个combined（combined和y表示同一个变量）
plt.plot(gggg, label='X')
plt.plot(p, label='Y')
plt.legend()
plt.title('X, Y Window')
plt.show()

#print("x = ")
#print(gggg)
#print("y = ")
#print(p)

# # Model Run

# In[8]:


float_precision_bits = 32
bits_in_MB = 8e6

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


# In[9]:


class Data(Dataset):
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

    def __len__(self):
        return len(self.x)


# In[10]:


from mvtae_model import MVTAEModel

hidden_vector_size = 64
hidden_alpha_size = 16
batch_size = 1024

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
#get_ipython().run_line_magic('tensorboard', '--port 6006 --logdir $tensorboard_folder')




model.fit(data_loader, epochs=10, start_epoch=0, verbose=True)





model.load_state_dict(torch.load('mvtae_model_best.pth'))


# # Hidden State Vector Visualisation           #显示部分输入数据的hidden state vector


model.eval()
zoom_lim = 10 # limit num of hidden vector examples to show
hidden_state_vector, decoder_output, alpha_output = model(from_numpy(tr_input_seq[:zoom_lim]).float())

# visualise hidden vector
for i in range(tr_input_seq[:zoom_lim].shape[0]):
    plt.plot(hidden_state_vector[0, i].detach().cpu())
plt.show()


# # Decoder Target Recreation                   #重构一个输入，然后与原始输入同时显示对比


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


# # Alpha Target Branch Prediction                 #使用tr_input_seq进行预测，与结果tr_data_windows_y进行比较。此处没有使用test数据，感觉不太好.


model.eval()
_,_ , alpha_output = model(from_numpy(tr_input_seq).float())
alpha_output = alpha_output.flatten().detach().cpu().numpy()

print('### Error/Accuracy Metrics ###')
print('MSE:\t', mean_squared_error(tr_data_windows_y, alpha_output))
print('MAE:\t', mean_absolute_error(tr_data_windows_y, alpha_output))
print('R²:\t', r2_score(tr_data_windows_y, alpha_output))

plt.plot(tr_data_windows_y, label='Target')
plt.plot(alpha_output, label='Prediction')
plt.legend()
plt.show()

##进行纯正的test，看看具体的情况





################################################################
# # Full Dataset Absolute Construction-Prediction Run



model.eval()
true = []
pred = []
for i in tqdm(range(data.shape[0])):
    if i < window_size:
        continue
    data_window = data[i-window_size:i]
    input_seq = np.zeros((1, window_size, len(features_x)))
    
    for j, feature in enumerate(features_x):
        _data_window, _, _= norm(data_window[feature])
        input_seq[0,:,j] = _data_window
    _, hi, lo = norm(data_window[feature_y])
    
    x_hidden_vector, decoder_output, alpha_output = model(from_numpy(input_seq).float())
    abs_pred = reverse_norm(alpha_output.squeeze().detach().cpu().numpy(), hi, lo)
    
    true.append(data[feature_y][i])
    pred.append(abs_pred)

plt.subplot(2,1,1)                                 #显示所有数据的原始数据和预测结果
plt.plot(true, label='Target')
plt.plot(pred, label='Prediction')
plt.title('Full Predictions De-Normalized')
plt.legend()

plt.subplot(2,1,2)                                 #显示前100个数据的原始数据和预测结果
zoom = 100
plt.plot(true[:zoom], label='Target')
plt.plot(pred[:zoom], label='Prediction')
plt.title('Full Predictions De-Normalized : Zoomed In')
plt.legend()
plt.show()


