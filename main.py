


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
from utils   import *

from mvtae_model import MVTAEModel

import matplotlib.pyplot as plt

import argparse

import os

print(torch.__version__)
print(torch.cuda.is_available())

def raw_data_schematic(data):
    idx_val_split = int(0.8 * data.shape[0])
    plt.plot(data['combined'], label='combined signal')
    plt.plot(data['sine_1'], label='sine_1 signal')
    plt.plot(data['sine_2'], label='sine_2 signal')
    plt.plot(data['noise'], alpha=0.5, label='noise signal')
    plt.vlines(idx_val_split, ymin=np.min(data['combined']), ymax=np.max(data['combined']), color='r', label='train/val split')
    plt.legend()
    plt.show()

    # # Transform & Normalize Data

    #显示第一个窗口的数据
    plt.plot(norm(data['sine_1'][:window_size])[0], label='sine_1')
    plt.plot(norm(data['sine_2'][:window_size])[0], label='sine_2')
    plt.plot(norm(data['combined'][:window_size])[0], label='combined')
    plt.legend()
    plt.show()

def norm_data_schematic(data, tr_input_seq, tr_data_windows_y):
    plt.subplot(2,2,1)
    for i in range(30):
        plt.plot(tr_input_seq[i,:,0])
    plt.title('Data Windows sine_1')

    plt.subplot(2,2,2)
    for i in range(30):
        plt.plot(tr_input_seq[i,:,1])
    plt.title('Data Windows sine_2')

    plt.subplot(2,1,2)              # test prediction Y with visual
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

# # Hidden State Vector Visualisation           #显示部分输入数据的hidden state vector

def hidden_state_verctor_visual(model, tr_input_seq):
    model.eval()
    zoom_lim = 10 # limit num of hidden vector examples to show
    hidden_state_vector, decoder_output, alpha_output = model(from_numpy(tr_input_seq[:zoom_lim]).float())
    # visualise hidden vector
    for i in range(tr_input_seq[:zoom_lim].shape[0]):
        plt.plot(hidden_state_vector[0, i].detach().cpu())
    plt.show()






# # Decoder Target Recreation                   #重构一个输入，然后与原始输入同时显示对比

def decoder_visual(model, tr_input_seq):
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

def alpha_train_visual(model, tr_input_seq, tr_data_windows_y):
    model.eval()
    _,_ , alpha_output = model(from_numpy(tr_input_seq).float())
    alpha_output = alpha_output.flatten().detach().cpu().numpy()
    plt.plot(tr_data_windows_y, label='Target')
    plt.plot(alpha_output, label='Prediction')
    plt.legend()
    plt.show()


##进行纯正的test，看看具体的情况
def alpha_test_visual(model, test_arr_x, test_arr_y):
    model.eval()
    _,_ , alpha_output = model(from_numpy(test_arr_x).float())
    alpha_output = alpha_output.flatten().detach().cpu().numpy()
    plt.plot(test_arr_y, label='Target')
    plt.plot(alpha_output, label='Prediction')
    plt.legend()
    plt.show()

def alpha_train_scores(model, tr_input_seq, tr_data_windows_y):
    model.eval()
    _,_ , alpha_output = model(from_numpy(tr_input_seq).float())
    alpha_output = alpha_output.flatten().detach().cpu().numpy()
    print('###  Train  Error/Accuracy Metrics ###')
    print('MSE:\t', mean_squared_error(tr_data_windows_y, alpha_output))
    print('MAE:\t', mean_absolute_error(tr_data_windows_y, alpha_output))
    print('R²:\t', r2_score(tr_data_windows_y, alpha_output))

def alpha_test_scores(model, test_arr_x, test_arr_y, test_arr_window):  #
    model.eval()
    _,_ , alpha_output = model(from_numpy(test_arr_x).float())
    alpha_output = alpha_output.flatten().detach().cpu().numpy()
    print('###  Test  Error/Accuracy Metrics ###')
    mse = mean_squared_error(test_arr_y, alpha_output)
    mae = mean_absolute_error(test_arr_y, alpha_output)
    r2  = r2_score(test_arr_y, alpha_output)
    print('MSE:\t', mse)
    print('MAE:\t', mae)
    print('R²:\t',  r2)


    #test_arr_x, test_arr_y, test_arr_window 维度应该是一样的
    #如果原始数据可能为0，就不能除以。股价应该没这个问题，除以后，看预测值的波动范围,与MSE，mae，r2比较，这些都是归一化后数据的指标
    list_y_raw = []
    list_y_pre = []

    for i in range(test_arr_window.shape[0]):
        win     = test_arr_window[i]
        y_raw, y_pre  = win.get_raw_data(alpha_output[i])
        list_y_raw.append(y_raw)
        list_y_pre.append(y_pre)


    deviationPercent = np.array(list_y_pre)/np.array(list_y_raw) - 1   #test_arr_y
    dev_per_mean = np.mean(deviationPercent)
    dev_per_std = np.std(deviationPercent)

    return mse, mae, r2, dev_per_mean, dev_per_std



def run_super_params():
    # Load Data

    fullpath = os.path.join('./data', id_stock)

    fullpath = fullpath + '.csv'

    data = pd.read_csv(fullpath, header=0, index_col=0)
    print(data)
    data.sort_values(by=FIELD_DATE, ascending=True, inplace=True)  #此处需要从前到后
    data = data.reset_index(drop=True)
    print(data)

    if pycharm_use:
        raw_data_schematic(data)

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
    test_arr_window = np.array(list_window[count_train:])   #还原预测结果使用

    tr_input_seq = train_arr_x
    tr_data_windows_y = train_arr_y
    #tr_data_size = tr_input_seq.shape[0]                   #3899

    if pycharm_use:
        norm_data_schematic(data, tr_input_seq, tr_data_windows_y)

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
                       dropout_p=dropout_p,
                       optim_lr=lr)

    print('-'*30)
    print('Data Batch Size:\t%.2fMB' % calc_input_memory((batch_size, tr_input_seq.shape[1], tr_input_seq.shape[2])))
    print('Model Size:\t\t%.2fMB' % calc_model_memory(model))
    print('Model Parameters:\t%d' % calc_model_params(model))
    print('-'*30)
    print('Data Size:\t\t', len(dataset))
    print('Batches per Epoch:\t', int(len(dataset)/tr_input_seq.shape[0]))

    best_epoch, best_loss = model.fit(data_loader, epochs=epochs_size, start_epoch=0, verbose=True)

    model.load_state_dict(torch.load('mvtae_model_best.pth'))

    if pycharm_use:
        hidden_state_verctor_visual(model, tr_input_seq)
        decoder_visual(model, tr_input_seq)
        alpha_train_visual(model, tr_input_seq, tr_data_windows_y)
        alpha_test_visual(model, test_arr_x, test_arr_y)

    alpha_train_scores(model, tr_input_seq, tr_data_windows_y)
    mse, mae, r2, dev_per_mean, dev_per_std = alpha_test_scores(model, test_arr_x, test_arr_y, test_arr_window)
    return mse, mae, r2, dev_per_mean, dev_per_std, best_epoch, best_loss


if __name__ == '__main__':
    print("main")
    parser = argparse.ArgumentParser()
    parser.add_argument('--pycharm', '--pycharm', required=False, default=0, help='use pycharm?')
    opt = vars(parser.parse_args())

    pycharm_use = opt["pycharm"]

    pycharm_use = 0

    #载入超参数配置文件，跑出来的结果写入文件

    df_super_params = pd.read_csv('super_params.csv', index_col = 'index', header=0)

    dic_super_params = df_super_params.to_dict('index')

    for index_sp in index_list_super_params:
        row = dic_super_params[index_sp]
        if not row:
            continue
        window_size        = int(row['window_size'])
        hidden_vector_size = int(row['hidden_vector_size'])
        hidden_alpha_size  = int(row['hidden_alpha_size'])
        batch_size         = int(row['batch_size'])
        weight_decoder     = int(row['weight_decoder'])
        epochs_size        = int(row['epochs_size'])
        dropout_p          = row['dropout_p']
        lr                 = row['lr']
        id_stock           = id_stock                                #不同的股票可以设置不同的超参数，自由组合

        mse, mae, r2, dev_per_mean, dev_per_std, best_epoch, best_loss = run_super_params()     #把结果写入文件

        if not os.path.exists("super_params_scores.csv"):
            df = pd.DataFrame(columns=['id_stock', 'index_sp', 'mse', 'mae', 'r2', 'dev_per_mean', 'dev_per_std', 'epochs', 'epoch_best', 'epoch_best_loss'])   #还需要记录预测与现实的关系
            df.to_csv("super_params_scores.csv", index = False)

        superParams = {}
        superParams['index_sp']            = index_sp
        superParams['mse']                 = mse
        superParams['mae']                 = mae
        superParams['r2']                  = r2
        superParams['dev_per_mean']        = dev_per_mean
        superParams['dev_per_std']         = dev_per_std
        superParams['epochs']              = epochs_size
        superParams['epoch_best']          = best_epoch
        superParams['epoch_best_loss']     = best_loss
        superParams['id_stock']            = id_stock
        dataframe = pd.read_csv("super_params_scores.csv")
        dataframe = dataframe.append([superParams], ignore_index = True)
        dataframe.to_csv("super_params_scores.csv", index = False)







