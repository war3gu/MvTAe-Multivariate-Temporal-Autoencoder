import math

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


from utils   import *
from defines import *

from mvtae_model import MVTAEModel
from mvtaeBinary_model import MVTAEBinaryModel

from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix

import matplotlib.pyplot as plt

import argparse

import os

import pandas_ta as ta

from expand_ta import *

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
    plt.plot(norm(data['sine_1'][:macro.window_size])[0], label='sine_1')
    plt.plot(norm(data['sine_2'][:macro.window_size])[0], label='sine_2')
    plt.plot(norm(data['combined'][:macro.window_size])[0], label='combined')
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
    xxxx = data[macroFeature.feature_y][:macro.window_size]             #取得第一个窗口的所有combined
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
    y_in = tr_input_seq[idx].reshape(1, macro.window_size, tr_input_seq.shape[2])
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
    
def alphaBinary_train_scores(model, tr_input_seq, tr_data_windows_y):
    
    print("train count {0}".format(tr_data_windows_y.shape[0]))
    
    model.eval()
    _,_ , alpha_output = model(from_numpy(tr_input_seq).float())
    alpha_output = alpha_output.flatten().detach().cpu().numpy()
    print('###  Train  Error/Accuracy Metrics ###')
    #print('MSE:\t', mean_squared_error(tr_data_windows_y, alpha_output))
    #print('MAE:\t', mean_absolute_error(tr_data_windows_y, alpha_output))
    #print('R²:\t', r2_score(tr_data_windows_y, alpha_output))

    lll = lambda x: int(x > 0.5)
    mmm = list(map(lll, alpha_output))
    cm = confusion_matrix(tr_data_windows_y, mmm, labels=[0,1])
    print(cm)
    #plot_confusion_matrix(cm, ['Down', 'Up'], normalize=True, title="Confusion Matrix")

    #true negatives is:math:`C_{0,0}`,
    #false negatives is :math:`C_{1,0}`,
    #true positives is:math:`C_{1,1}`
    #false positives is :math:`C_{0,1}`.
    tn, fp, fn, tp = cm.ravel()
    
    if tp+fn == 0:    #修正一下，防止除以0
        tp = 1
        fn = 1
        print("fix tp fn")
    if tn+fp == 0:
        tn = 1
        fp = 1
        print("fix tn fp")

    per_pp = tp/(tp+fn)   #TPR
    per_nn = tn/(tn+fp)   #TNR
    per_corr = (tp+tn)/(tn+fp+fn+tp)
    print("train score")
    print(per_pp)
    print(per_nn)
    print(per_corr)

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

    posi_posi_count = 0
    nega_nega_count = 0
    posi_count = 0
    nega_count = 0

    for i in range(test_arr_window.shape[0]):
        win     = test_arr_window[i]
        y_raw, y_pre, y_raw_1 = win.get_raw_data(alpha_output[i])
        list_y_raw.append(y_raw)
        list_y_pre.append(y_pre)
        if y_raw > y_raw_1:
            posi_count += 1
            if y_pre > y_raw_1:
                posi_posi_count += 1
        else:
            nega_count += 1
            if y_pre <= y_raw_1:
                nega_nega_count += 1

    per_pp = posi_posi_count/posi_count
    per_nn = nega_nega_count/nega_count
    per_corr = (posi_posi_count+nega_nega_count)/(posi_count+nega_count)

    deviationPercent = np.array(list_y_pre)/np.array(list_y_raw) - 1   #test_arr_y
    dev_per_mean = np.mean(deviationPercent)
    dev_per_std = np.std(deviationPercent)

    return mse, mae, r2, dev_per_mean, dev_per_std, per_pp, per_nn, per_corr

def alphaBinary_test_scores(model, test_arr_x, test_arr_y, test_arr_window):  #
    
    print("test count {0}".format(test_arr_y.shape[0]))
    
    model.eval()
    _,_ , alpha_output = model(from_numpy(test_arr_x).float())
    alpha_output = alpha_output.flatten().detach().cpu().numpy()

    lll = lambda x: int(x > 0.5)
    mmm = list(map(lll, alpha_output))
    cm = confusion_matrix(test_arr_y, mmm, labels=[0,1])
    print(cm)
    #plot_confusion_matrix(cm, ['Down', 'Up'], normalize=True, title="Confusion Matrix")

    tn, fp, fn, tp = cm.ravel()
    
    if tp+fn == 0:    #修正一下，防止除以0
        tp = 1
        fn = 1
        print("fix tp fn")
    if tn+fp == 0:
        tn = 1
        fp = 1
        print("fix tn fp")

    per_pp = tp/(tp+fn)   #TPR
    per_nn = tn/(tn+fp)   #TNR
    per_corr = (tp+tn)/(tn+fp+fn+tp)

    print("test score")
    print(per_pp)
    print(per_nn)
    print(per_corr)

    return 0, 0, 0, 0, 0, per_pp, per_nn, per_corr



def run_super_params():
    # Load Data

    fullpath = os.path.join('./data', macro.id_stock)

    fullpath = fullpath + '.csv'

    data = pd.read_csv(fullpath, header=0, index_col=0)
    #print(data)
    data.sort_values(by=FIELD_DATE, ascending=True, inplace=True)  #此处需要从前到后
    data = data.reset_index(drop=True)
    #print(data)

    if pycharm_use:
        raw_data_schematic(data)

    raw_data_size = data.shape[0]
    index_window_start = 0
    index_window_end = index_window_start + macro.window_size
    list_window = []
    list_x = []
    list_y = []

    while index_window_end < raw_data_size:
        data_x_temp = data[index_window_start:index_window_end]
        data_y_temp = data.iloc[index_window_end]

        oneWin = window(data_x_temp, data_y_temp)
        oneWin.norm()
        ret_x_df, ret_y_df = oneWin.get_norm_data_frame()

        if binary_run:
            ret_x_arr, ret_y_arr = oneWin.get_up_down_array()
        else:
            ret_x_arr, ret_y_arr = oneWin.get_norm_data_array()

        list_x.append(ret_x_arr)
        list_y.append(ret_y_arr)
        list_window.append(oneWin)
        index_window_start = index_window_start + step_size
        index_window_end = index_window_start + macro.window_size

    count_window = len(list_x)
    count_train = int(np.ceil(count_window * macro.split_ratio))
    #print("train count")
    #print(count_train)
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
        batch_size=macro.batch_size,
        shuffle = True
    )

    model = None
    if binary_run:
        model = MVTAEBinaryModel(model_save_path='./',
                                 seq_len=tr_input_seq.shape[1],
                                 in_data_dims=tr_input_seq.shape[2],
                                 out_data_dims=tr_input_seq.shape[2],
                                 model_name='mvtae_model',
                                 hidden_vector_size=macro.hidden_vector_size,
                                 hidden_alpha_size=macro.hidden_alpha_size,
                                 dropout_p=macro.dropout_p,
                                 optim_lr=macro.lr,
                                 optim_weight_decay=macro.weight_decay)
    else:
        model = MVTAEModel(model_save_path='./',
                           seq_len=tr_input_seq.shape[1],
                           in_data_dims=tr_input_seq.shape[2],
                           out_data_dims=tr_input_seq.shape[2],
                           model_name='mvtae_model',
                           hidden_vector_size=macro.hidden_vector_size,
                           hidden_alpha_size=macro.hidden_alpha_size,
                           dropout_p=macro.dropout_p,
                           optim_lr=macro.lr,
                           optim_weight_decay=macro.weight_decay)



    print('-'*30)
    print('Data Batch Size:\t%.2fMB' % calc_input_memory((macro.batch_size, tr_input_seq.shape[1], tr_input_seq.shape[2])))
    print('Model Size:\t\t%.2fMB' % calc_model_memory(model))
    print('Model Parameters:\t%d' % calc_model_params(model))
    print('-'*30)
    print('Data Size:\t\t', len(dataset))
    #print(tr_input_seq.shape)
    #print('Batches per Epoch:\t', int(len(dataset)/tr_input_seq.shape[0]))    #这个好像错了

    best_epoch, best_loss = model.fit(data_loader, epochs=macro.epochs_size, start_epoch=0, verbose=True)

    model.load_state_dict(torch.load('mvtae_model_best.pth'))

    if pycharm_use:
        hidden_state_verctor_visual(model, tr_input_seq)
        decoder_visual(model, tr_input_seq)
        alpha_train_visual(model, tr_input_seq, tr_data_windows_y)
        alpha_test_visual(model, test_arr_x, test_arr_y)

    if binary_run:
        alphaBinary_train_scores(model, tr_input_seq, tr_data_windows_y)
        mse, mae, r2, dev_per_mean, dev_per_std, per_pp, per_nn, per_corr = alphaBinary_test_scores(model, test_arr_x, test_arr_y, test_arr_window)
        return mse, mae, r2, dev_per_mean, dev_per_std, per_pp, per_nn, per_corr, best_epoch, best_loss
    else:
        alpha_train_scores(model, tr_input_seq, tr_data_windows_y)
        mse, mae, r2, dev_per_mean, dev_per_std, per_pp, per_nn, per_corr = alpha_test_scores(model, test_arr_x, test_arr_y, test_arr_window)
        return mse, mae, r2, dev_per_mean, dev_per_std, per_pp, per_nn, per_corr, best_epoch, best_loss

def run_super_params_minute(isFive, row_ta):
    # Load Data

    path = "./line_convert"

    if isFive == True:
        path = os.path.join(path, "fzline")
    else:
        path = os.path.join(path, "minline")

    ssss = macro.id_stock.split('.')

    id_stock = ssss[1] + ssss[0]

    fullpath = os.path.join(path, id_stock)

    fullpath = fullpath + '.csv'

    data = pd.read_csv(fullpath, header=None, index_col=0)   #需要异常数据处理,有的为0

    fill_zero_last(data, 1)
    fill_zero_last(data, 2)
    fill_zero_last(data, 3)
    fill_zero_last(data, 4)
    fill_zero_last(data, 5)
    fill_zero_last(data, 6)

    data = expand_data(data, row_ta)                     #丰富数据,同时把列名给改了

    #可能还需要对close取若干天的均线

    '''
    if macro.log_price:
        print("start log data")
        log_data(data, 1)
        log_data(data, 2)
        log_data(data, 3)
        log_data(data, 4)
        log_data(data, 5)
        log_data(data, 6)
    '''

    #print(data)
    #data.sort_values(by=FIELD_DATE, ascending=True, inplace=True)  #此处需要从前到后
    #data = data.reset_index(drop=True)
    #print(data)

    if pycharm_use:
        raw_data_schematic(data)

    raw_data_size = data.shape[0]
    index_window_start = 0
    index_window_end = index_window_start + macro.window_size
    list_window = []
    list_x = []
    list_y = []

    if macro.data_size != 0:
        raw_data_size = macro.data_size

    while index_window_end < raw_data_size:                    #此处分析没有因为隔天把数据截断
        data_x_temp = data[index_window_start:index_window_end]
        data_y_temp = data.iloc[index_window_end]

        oneWin = window(data_x_temp, data_y_temp)
        oneWin.norm()
        ret_x_df, ret_y_df = oneWin.get_norm_data_frame()

        if binary_run:
            ret_x_arr, ret_y_arr = oneWin.get_up_down_array()   #window_size为0，ret_x_arr全是1，样本就集中在一个点，这样是不行的
        else:
            ret_x_arr, ret_y_arr = oneWin.get_norm_data_array()

        list_x.append(ret_x_arr)
        list_y.append(ret_y_arr)
        list_window.append(oneWin)
        index_window_start = index_window_start + step_size
        index_window_end = index_window_start + macro.window_size

    count_window = len(list_x)
    count_train = int(np.ceil(count_window * macro.split_ratio))
    #print("train count")
    #print(count_train)
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
        batch_size=macro.batch_size,
        shuffle = True
    )

    model = None
    if binary_run:
        model = MVTAEBinaryModel(model_save_path='./',
                                 seq_len=tr_input_seq.shape[1],
                                 in_data_dims=tr_input_seq.shape[2],
                                 out_data_dims=tr_input_seq.shape[2],
                                 model_name='mvtae_model',
                                 hidden_vector_size=macro.hidden_vector_size,
                                 hidden_alpha_size=macro.hidden_alpha_size,
                                 dropout_p=macro.dropout_p,
                                 optim_lr=macro.lr,
                                 optim_weight_decay=macro.weight_decay)
    else:
        model = MVTAEModel(model_save_path='./',
                           seq_len=tr_input_seq.shape[1],
                           in_data_dims=tr_input_seq.shape[2],
                           out_data_dims=tr_input_seq.shape[2],
                           model_name='mvtae_model',
                           hidden_vector_size=macro.hidden_vector_size,
                           hidden_alpha_size=macro.hidden_alpha_size,
                           dropout_p=macro.dropout_p,
                           optim_lr=macro.lr,
                           optim_weight_decay=macro.weight_decay)



    print('-'*30)
    print('Data Batch Size:\t%.2fMB' % calc_input_memory((macro.batch_size, tr_input_seq.shape[1], tr_input_seq.shape[2])))
    print('Model Size:\t\t%.2fMB' % calc_model_memory(model))
    print('Model Parameters:\t%d' % calc_model_params(model))
    print('-'*30)
    print('Data Size:\t\t', len(dataset))
    #print(tr_input_seq.shape)
    #print('Batches per Epoch:\t', int(len(dataset)/tr_input_seq.shape[0]))    #这个好像错了

    best_epoch, best_loss = model.fit(data_loader, epochs=macro.epochs_size, start_epoch=0, verbose=True)

    model.load_state_dict(torch.load('mvtae_model_best.pth'))

    if pycharm_use:
        hidden_state_verctor_visual(model, tr_input_seq)
        decoder_visual(model, tr_input_seq)
        alpha_train_visual(model, tr_input_seq, tr_data_windows_y)
        alpha_test_visual(model, test_arr_x, test_arr_y)

    if binary_run:
        alphaBinary_train_scores(model, tr_input_seq, tr_data_windows_y)
        mse, mae, r2, dev_per_mean, dev_per_std, per_pp, per_nn, per_corr = alphaBinary_test_scores(model, test_arr_x, test_arr_y, test_arr_window)
        return mse, mae, r2, dev_per_mean, dev_per_std, per_pp, per_nn, per_corr, best_epoch, best_loss
    else:
        alpha_train_scores(model, tr_input_seq, tr_data_windows_y)
        mse, mae, r2, dev_per_mean, dev_per_std, per_pp, per_nn, per_corr = alpha_test_scores(model, test_arr_x, test_arr_y, test_arr_window)
        return mse, mae, r2, dev_per_mean, dev_per_std, per_pp, per_nn, per_corr, best_epoch, best_loss






def run_stock(id_stock, dic_super_params, dic_super_ta_params):
    for index_sp in index_list_super_params:
        row = dic_super_params[index_sp]
        if not row:
            continue
        print("start run sp {0}".format(index_sp))

        for index_sp_ta in index_list_super_ta_params:
            row_ta = dic_super_ta_params[index_sp_ta]
            if not row_ta:
                continue

            print("start run sp_ta {0}".format(index_sp_ta))
            macro.window_size        = int(row['window_size'])
            macro.hidden_vector_size = int(row['hidden_vector_size'])
            print("hidden_vector_size {0}".format(macro.hidden_vector_size))
            macro.hidden_alpha_size  = int(row['hidden_alpha_size'])
            macro.batch_size         = int(row['batch_size'])
            macro.weight_decoder     = float(row['weight_decoder'])
            macro.weight_alpha       = float(row['weight_alpha'])
            macro.epochs_size        = int(row['epochs_size'])
            macro.dropout_p          = row['dropout_p']
            macro.lr                 = row['lr']
            macro.weight_decay       = row['weight_decay']
            macro.split_ratio        = float(row['split_ratio'])
            macro.data_size          = int(row['data_size'])


            macro.id_stock           = id_stock                              #不同的股票可以设置不同的超参数，自由组合(暂时先这样跑)


            if time_model == 'day':
                mse, mae, r2, dev_per_mean, dev_per_std, per_pp, per_nn, per_corr, best_epoch, best_loss = run_super_params()     #把结果写入文件
            elif time_model == 'min5':
                mse, mae, r2, dev_per_mean, dev_per_std, per_pp, per_nn, per_corr, best_epoch, best_loss = run_super_params_minute(True, row_ta)
            else:
                mse, mae, r2, dev_per_mean, dev_per_std, per_pp, per_nn, per_corr, best_epoch, best_loss = run_super_params_minute(False, row_ta)

            #print(macro.weight_decoder)
            #print(macro.epochs_size)
            if not os.path.exists("super_params_scores.csv"):
                df = pd.DataFrame(columns=['id_stock', 'index_sp', 'index_sp_ta', 'mse', 'mae', 'r2', 'dev_per_mean', 'dev_per_std', 'per_pp', 'per_nn', 'per_corr', 'epochs', 'epoch_best', 'epoch_best_loss'])   #还需要记录预测与现实的关系
                df.to_csv("super_params_scores.csv", index = False)

            superParamsScores = {}
            superParamsScores['index_sp']            = index_sp
            superParamsScores['index_sp_ta']         = index_sp_ta
            superParamsScores['mse']                 = round(mse, 4)
            superParamsScores['mae']                 = round(mae, 4)
            superParamsScores['r2']                  = round(r2, 4)
            superParamsScores['dev_per_mean']        = round(dev_per_mean, 4)
            superParamsScores['dev_per_std']         = round(dev_per_std, 4)
            superParamsScores['per_pp']              = round(per_pp, 4)
            superParamsScores['per_nn']              = round(per_nn, 4)
            superParamsScores['per_corr']            = round(per_corr, 4)
            superParamsScores['epochs']              = macro.epochs_size        #
            superParamsScores['epoch_best']          = best_epoch               #
            superParamsScores['epoch_best_loss']     = round(best_loss, 4)
            superParamsScores['id_stock']            = macro.id_stock           #
            dataframe = pd.read_csv("super_params_scores.csv")
            dataframe = dataframe.append([superParamsScores], ignore_index = True)
            dataframe.to_csv("super_params_scores.csv", index = False)
            print("end run {0}".format(index_sp))


if __name__ == '__main__':
    print("main")
    parser = argparse.ArgumentParser()
    parser.add_argument('--pycharm', '--pycharm', required=False, default=0, help='use pycharm?')
    opt = vars(parser.parse_args())

    pycharm_use = opt["pycharm"]

    pycharm_use = 0

    binary_run = False

    if type_model == "binary":
        binary_run = True

    #载入超参数配置文件，跑出来的结果写入文件

    df_super_params = pd.read_csv('super_params.csv', index_col = 'index', header=0)

    dic_super_params = df_super_params.to_dict('index')

    df_super_ta_params = pd.read_csv('super_ta_params.csv', index_col = 'index', header=0)

    dic_super_ta_params = df_super_ta_params.to_dict('index')

    

    print("444444")
    
    run_stock("600507.sh", dic_super_params, dic_super_ta_params)
    #run_stock("000002.sz", dic_super_params, dic_super_ta_params)
    #run_stock("000333.sz", dic_super_params, dic_super_ta_params)
    
 
    '''
    stocks_list = pd.read_csv('stocks_list.csv',)
    print("88888")
    for i in range(640,680):                     
        stock_df = stocks_list.iloc[i]
        ts_code = stock_df.ts_code.lower()
        print("index {0}".format(i))
        print("run {0}".format(ts_code))
        fullPath = os.path.join(data_folder, ts_code)
        fullName = '{}.csv'.format(fullPath)
        if os.path.isfile(fullName):
            run_stock(ts_code, dic_super_params, dic_super_ta_params)
    '''
    
    '''
    for stock in stocks_list:
        print("here")
        run_stock(stock, dic_super_params, dic_super_ta_params)
    '''
   











