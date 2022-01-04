__author__ = "Jakob Aungiers"
__copyright__ = "Copyright 2020, Jakob Aungiers // Altum Intelligence"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "research@altumintelligence.com"

import os
from tqdm import tqdm
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import math

import sys
#from defines import *

from utils   import *

class MVTAEModel(nn.Module):
    
    def __init__(self,
                 model_save_path,
                 seq_len, in_data_dims,
                 out_data_dims,
                 model_name,
                 hidden_vector_size,
                 hidden_alpha_size,
                 dropout_p,
                 optim_lr,
                 optim_weight_decay):
        super(MVTAEModel, self).__init__()
        self.seq_len = seq_len
        self.in_data_dims = in_data_dims
        self.out_data_dims = out_data_dims
        self.model_save_path = model_save_path
        
        self.model_name = model_name
        self.best_loss = 1e10
        self.best_epoch = None
        if model_save_path:
            self.tb_writer = SummaryWriter(log_dir='./tensorboard')
        self.device = 'cuda'
        print('Using', self.device)
        if self.device == 'cuda':
            print('Using GPU:', torch.cuda.get_device_name(self.device))
            
        self.build_model(hidden_vector_size, hidden_alpha_size, dropout_p, optim_lr, optim_weight_decay)
        self.to(self.device)
        self.eval()                                        #Sets the module in evaluation mode
        
    def build_model(self, hidden_vector_size, hidden_alpha_size, dropout_p, optim_lr, optim_weight_decay):
        self.dropout = nn.Dropout(p=dropout_p)
        self.encoder = nn.LSTM(input_size=self.in_data_dims, hidden_size=hidden_vector_size, batch_first=True)
        self.decoder = nn.LSTM(input_size=self.encoder.hidden_size, hidden_size=self.encoder.hidden_size, batch_first=False)
        self.decoder_output = nn.Linear(self.encoder.hidden_size, self.out_data_dims)

        self.alpha_hidden_1 = nn.Linear(self.encoder.hidden_size, hidden_alpha_size)
        self.alpha_hidden_2 = nn.Linear(hidden_alpha_size, hidden_alpha_size)
        self.alpha_out = nn.Linear(hidden_alpha_size, 1)
        
        self.loss_decoder = nn.MSELoss()
        self.loss_alpha = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=optim_lr, weight_decay=optim_weight_decay)    #初始化优化器，将模型参数填入
        print("normal")
        print("打印参数")
        for name, param in self.named_parameters():
            print(name, "  ", param.requires_grad)
        print("打印参数结束")
        
    def forward(self, x):
        x = x.to(self.device)                          #encoder的batch_first为true,x的shape为（batch_size，seq_len，in_data_dims）,两者需要吻合！！！

        # Encoder
        encoder_out, encoder_hidden = self.encoder(x)  #因为encoder的batch_first为true，encoder_out的shape为（batch_size，seq_len，hidden_vector_size）
        hidden_state_vector = encoder_hidden[0]        #encoder_hidden 就是hn和cn.注意，shape没变，为（1，batch_size，hidden_vector_size），不受batch_first影响
        
        # Decoder
        encoder_hidden_dropout = self.dropout(hidden_state_vector)
        rrrrrr = encoder_hidden_dropout.repeat(self.seq_len, 1, 1)    #encoder输出的hidden vector，在dropout之后，对第0维进行扩展，方便输入decoder
        decoder_out, decoder_hidden = self.decoder(rrrrrr)            #输入decoder，在tensorboard中显示为4tensor，实际就一个变量
        tttttt = decoder_out.transpose(0,1)                           #交换第0维和第1维,把batch放在最前面
        decoder_output = self.decoder_output(tttttt)                  #进行最后的线性变化，还原回最初的原始数据

        # Alpha
        alpha_hidden_1 = F.relu(self.alpha_hidden_1(hidden_state_vector))      #线性变换，relu
        alpha_hidden_1_dropout = self.dropout(alpha_hidden_1)                  #dropout
        alpha_hidden_2 = F.relu(self.alpha_hidden_2(alpha_hidden_1_dropout))   #线性变换，relu
        oooooo = self.alpha_out(alpha_hidden_2)                                #线性变换.(此处之前是alpha_hidden_1，感觉简单了点，都可以试试)
        alpha_output = oooooo.squeeze()                                        #移除长度为1的维度，实际剩余就是长度为batch_size的一维数组，就是预测结果

        return hidden_state_vector, decoder_output, alpha_output

    def predict(self, x):
        return self(x)
    
    def fit(self, data_loader, epochs, start_epoch=0, verbose=False):
        ddq = iter(data_loader)                                                  #迭代器
        images, labels = next(ddq)                                               #取得第一组数据
        #print(images)
        #print(labels)
        #images1, labels1 = next(ddq)
        #print(images1)
        #print(labels1)
        self.tb_writer.add_graph(self, input_to_model=images, verbose=False)     #就是这样的，没错！！！！
        self.tb_writer.close()

        best_epoch, best_loss = self.fitOnce(data_loader, epochs=epochs, start_epoch=start_epoch, verbose=verbose)
        return best_epoch, best_loss

    '''
        print("start train decoder")
        best_epoch, best_loss = self.fit_weight(data_loader, epochs=500000, start_epoch=start_epoch, verbose=verbose, decoder_weight=1, alpha_weight=0)
        self.best_loss = 1e10
        self.best_epoch = None
        self.encoder.requires_grad_(False)              #锁定encoder，不用再训练了
        print("start train alpha")
        best_epoch, best_loss = self.fit_weight(data_loader, epochs=500000, start_epoch=start_epoch, verbose=verbose, decoder_weight=0, alpha_weight=1)
        print("train end")
        return best_epoch, best_loss
    '''


    def fitOnce(self, data_loader, epochs, start_epoch=0, verbose=False):
        with open('loss_value.log', 'w') as flog:
            flog.write('Timestamp,Epoch,Loss\n')

        #epochs_count = epochs - start_epoch
        #mul = macro.weight_decoder/0.001
        #lll = math.log(mul)
        #ccc = -(lll/epochs_count)
        #eee = math.exp(ccc)
        #weight_last = macro.weight_decoder

        for i in tqdm(range(start_epoch, epochs), disable=not verbose):
            self.train()                                                          # set model to training mode
            loss_value = 0
            #weight_last = weight_last * eee
            for x_batch, y_batch in data_loader:
                x = x_batch.to(self.device)
                x_inv = x.flip(1)                                                 #翻转第一维，也就是seq_len那一维
                y = y_batch.to(self.device)

                self.optimizer.zero_grad()                                        #所有tensor的梯度归零
                hidden_state_vector, decoder_output, alpha_output = self(x)       #调用forward（batch_size为第0维）

                loss_decoder = self.loss_decoder(decoder_output, x_inv)
                loss_alpha = self.loss_alpha(alpha_output, y)
                loss = loss_decoder*macro.weight_decoder + loss_alpha             #此处可以加一个超参数权重，需要寻找使alpha test loss最小的值

                loss_value = loss_value + loss.item()

                loss.backward()                                                   #计算梯度并保存在tensor中

                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.5)            #对梯度进行裁剪，防止梯度爆炸
                self.optimizer.step()                                             #根据梯度更新参数

            if loss_value < self.best_loss:
                self.best_loss = loss_value
                self.best_epoch = i
                torch.save(self.state_dict(), os.path.join(self.model_save_path, '{0}_best.pth'.format(self.model_name)))
            self.tb_writer.add_scalar('loss_value', loss_value, i)
            with open('loss_value.log', 'a') as flog:
                flog.write('{0},{1},{2},{3},{4}\n'.format(datetime.utcnow(), i, loss_value, loss_decoder, loss_alpha))

            epoch_pass = i-self.best_epoch
            if epoch_pass > 200:               #过了200epoch还没有提升
                break

        print('Best epoch: {0} | loss {1}'.format(self.best_epoch, self.best_loss))
        sum_best_loss = self.best_loss          #一般性的方法是转化为ndarray再取值
        return int(self.best_epoch), sum_best_loss

    def fit_weight(self, data_loader, epochs, start_epoch=0, verbose=False, decoder_weight=1, alpha_weight=0):
        flog_name = 'loss_{0}.log'.format(decoder_weight)
        with open(flog_name, 'w') as flog:
            flog.write('Timestamp,Epoch,Loss\n')

        for i in tqdm(range(start_epoch, epochs), disable=not verbose):
            self.train()                                                         # set model to training mode

            for x_batch, y_batch in data_loader:
                x = x_batch.to(self.device)
                x_inv = x.flip(1)                                                 #翻转第一维，也就是seq_len那一维
                y = y_batch.to(self.device)

                self.optimizer.zero_grad()                                        #所有tensor的梯度归零
                hidden_state_vector, decoder_output, alpha_output = self(x)       #调用forward（batch_size为第0维）

                loss_decoder = self.loss_decoder(decoder_output, x_inv)
                loss_alpha = self.loss_alpha(alpha_output, y)
                loss = loss_decoder*decoder_weight + loss_alpha*alpha_weight      #此处可以加一个超参数权重，需要寻找使alpha test loss最小的值

                loss.backward()                                                   #计算梯度并保存在tensor中

                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.5)            #对梯度进行裁剪，防止梯度爆炸
                self.optimizer.step()                                             #根据梯度更新参数

            if loss < self.best_loss:
                self.best_loss = loss
                self.best_epoch = i
                torch.save(self.state_dict(), os.path.join(self.model_save_path, '{0}_best_{1}.pth'.format(self.model_name, decoder_weight)))
            self.tb_writer.add_scalar('loss', loss, i)
            with open(flog_name, 'a') as flog:
                flog.write('{0},{1},{2},{3},{4}\n'.format(datetime.utcnow(), i, loss, loss_decoder, loss_alpha))

            epoch_pass = i-self.best_epoch
            if epoch_pass > 1000:               #过了1000epoch还没有提升
                break
        print('Best epoch: {0} | loss {1} | decoder_weight {2}'.format(self.best_epoch, self.best_loss, decoder_weight))
        sum_best_loss = self.best_loss.item()         #一般性的方法是转化为ndarray再取值
        return int(self.best_epoch), sum_best_loss