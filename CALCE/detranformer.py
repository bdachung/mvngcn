import numpy as np
import random
import math
import os
import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
import transformers

from math import sqrt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import functools
from sklearn.model_selection import train_test_split

from utils import *
from cs_interface import NestedModel

class Autoencoder(nn.Module):
    def __init__(self, input_size=16, hidden_dim=8, noise_level=0.01):
        super(Autoencoder, self).__init__()
        self.input_size, self.hidden_dim, self.noise_level = input_size, hidden_dim, noise_level
        self.fc1 = nn.Linear(self.input_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.input_size)
        
    def encoder(self, x):
        x = self.fc1(x)
        h1 = F.relu(x)
        return h1
    
    def mask(self, x):
        corrupted_x = x + self.noise_level * torch.randn_like(x)
        return corrupted_x
    
    def decoder(self, x):
        h2 = self.fc2(x)
        return h2
    
    def forward(self, x):
        out = self.mask(x)
        encode = self.encoder(out)
        decode = self.decoder(encode)
        return encode, decode

class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, d_model, dropout=0.0, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        self.P = torch.zeros(())
        
        # Create a long enough P
        self.P = torch.zeros((1, max_len, d_model))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, d_model, 2, dtype=torch.float32) / d_model)
        
        self.P[:, :, 0::2] = torch.sin(X[:,:d_model//2 + d_model%2])
        self.P[:, :, 1::2] = torch.cos(X[:,:d_model//2])

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class Net(nn.Module):
    def __init__(self, feature_size=16, hidden_dim=32, num_layers=1, nhead=8, dropout=0.0, noise_level=0.01):
        super(Net, self).__init__()
        self.auto_hidden = int(feature_size/2)
        input_size = self.auto_hidden 
        self.pos = PositionalEncoding(d_model=input_size, max_len=input_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        self.cell = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.linear = nn.Linear(input_size, 1)
        self.autoencoder = Autoencoder(input_size=feature_size, hidden_dim=self.auto_hidden, noise_level=noise_level)
 
    def forward(self, x): 
        batch_size, feature_num, feature_size  = x.shape 
        encode, decode = self.autoencoder(x.reshape(batch_size, -1))# batch_size*seq_len
        out = encode.reshape(batch_size, -1, self.auto_hidden)
        out = self.pos(out)
        out = out.reshape(1, batch_size, -1) # (1, batch_size, feature_size)
        out = self.cell(out)  
        out = out.reshape(batch_size, -1) # (batch_size, hidden_dim)
        out = self.linear(out)            # out shape: (batch_size, 1)
        
        return out, decode

class DeTransformer(NestedModel):
    def build_model(self, feature_size, hidden_dim, num_layers, nhead, dropout, noise_level):
        return Net(feature_size=feature_size, hidden_dim=hidden_dim, num_layers=num_layers, nhead=nhead, dropout=dropout, noise_level=noise_level)

    def extract_batter_features(self, concat_df, global_df, battery_nb, window_size=8, prediction_interval=1, extract_points_len=11, multichannel_features=['voltage','current','resistance'], global_seq_features = ['x5', 'rest_period', 'capacity'], key='cycle', label='capacity'):
        # choose only 1 battery at a time
        #   df_copy = concat_df[battery_nb].copy()
        df_copy = concat_df.copy()
        # rest time of each charge cycle
        #   global_df_copy = global_df[battery_nb].copy()
        global_df_copy = global_df.copy()

        capacity = df_copy.groupby([key]).mean()[label].to_numpy().reshape(-1, 1)

        global_feature = global_df_copy[global_seq_features].to_numpy()

        multi_channel_feature = df_copy[multichannel_features].to_numpy().reshape(-1, extract_points_len, len(multichannel_features))

        # sliding window
        def sliding_window(window_length, input_matrix):
            nb_items = input_matrix.shape[0]
            sub_window = np.array([np.arange(start=x, stop=x+window_length, step=1) for x in range(nb_items-window_length)])
            return input_matrix[sub_window]

        slided_mc_feature = sliding_window(window_size, multi_channel_feature)
        slided_glb_feature = sliding_window(window_size, global_feature)
        #   print(slided_mc_feature.shape)
        slided_capacity = capacity[window_size:]
        #   print(slided_capacity.shape)

        # shifting forecast interval
        input = slided_glb_feature[:-prediction_interval]
        output = slided_capacity[prediction_interval:]

        return input, output

    def cross_validation(self, batteries, concat_df, global_df, extract_points_len=11, multichannel_features=['voltage','current','resistance'], window_size=8, prediction_interval=1, global_seq_features = ['x5', 'rest_period', 'capacity'], key='cycle', label='capacity', validation=None, random_state=7, shuffle=False):
    # last pin is the test set
        all_batteries = [self.extract_batter_features(concat_df[battery_nb].copy(), global_df[battery_nb].copy(), battery_nb, window_size=window_size, prediction_interval=prediction_interval, extract_points_len=extract_points_len, multichannel_features=multichannel_features, global_seq_features=global_seq_features, key=key, label=label) for battery_nb in batteries]

        # cross validation
        train_set = all_batteries[0:-1]
        val_set = all_batteries[-1]
        test_set = all_batteries[-1]

        glb_train = functools.reduce(lambda a,b: np.concatenate((a,b), axis=0), [x for x,_ in train_set]).reshape(-1, window_size, len(global_seq_features))

        Y_train = functools.reduce(lambda a,b: np.concatenate((a,b), axis=0), [y for _,y in train_set])
        Y_train = Y_train.reshape(-1, 1)

        if validation is None:
            X_train = glb_train

            glb_val, Y_val = val_set
            X_val = glb_val

        else:
            glb_train, glb_val, Y_train, Y_val = train_test_split(glb_train, Y_train, test_size=validation, random_state=random_state, shuffle=shuffle)
            X_train = glb_train
            X_val = glb_val

        glb_test, Y_test = test_set
        X_test = glb_test

        return X_train, Y_train, X_val, Y_val, X_test,Y_test

    def train(self, model, X_train, Y_train, X_val, Y_val, out_dir, EPOCH=1000, seed=0, dropout=0., LR=0.001, weight_decay=0, device='cpu', alpha=0.05, feature_size=10, clipping_value=None,is_load_weights=False):
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        loss_list, y_ = [0], [] 
        rmse_list, mape_list = [], []
        best_rmse = None
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        if is_load_weights: 
            if torch.__version__.split('+')[0] >= '1.6.0':
                model.load_state_dict(torch.load('initial_weights/model_NASA.pth')) 
            else:
                model.load_state_dict(torch.load('initial_weights/model_NASA_1.5.0.pth'))

        for epoch in range(EPOCH) :
            model.train()
            X_local = X_train
            X_local = np.array(X_local).astype(np.float32)
            Y = np.array(Y_train).astype(np.float32)
            X_local = torch.from_numpy(X_local).to(device)
            X_local = torch.reshape(X_local, (X_local.size(0),1,-1))
    #         print(X_local.shape)
            Y = torch.from_numpy(Y).to(device)
            output, decode = model(X_local)
            output = output.reshape(-1, 1)
            loss = criterion(output, Y) + alpha * criterion(decode, X_local.reshape(-1, feature_size))
            optimizer.zero_grad()
            loss.backward()
            if clipping_value:
                nn.utils.clip_grad_norm(model.parameters(), clipping_value)
            optimizer.step()

            if epoch%10==0 or epoch==(EPOCH-1):
                point_list = []
                x_local = X_val
                x_local = np.array(x_local).astype(np.float32)
                x_local = torch.from_numpy(x_local).to(device)
                x_local = torch.reshape(x_local, (x_local.size(0),1,-1))
                model.eval()
                with torch.no_grad():
                    pred, _ = model(x_local)
    #                 print(pred)
                point_list = list(pred[:,0].cpu().detach().numpy())
                y_.append(point_list)
                loss_list.append(loss)
                mape, rmse = evaluation(y_test=denormalize(Y_val).flatten(), y_predict=denormalize(np.array(y_[-1])))
                mape_list.append(mape)
                rmse_list.append(rmse)
                if best_rmse == None:
                    best_rmse = {}
                    # best_rmse['name'] = 'checkpoint-' + str(epoch) + '.pt'
                    best_rmse['name'] = 'checkpoint.pt'
                    best_rmse['value'] = rmse
                    torch.save(model, os.path.join(out_dir,best_rmse['name']))
                else:
                    if rmse < best_rmse['value']:
                        os.remove(os.path.join(out_dir,best_rmse['name']))
                        best_rmse['name'] = 'checkpoint.pt'
                        best_rmse['value'] = rmse
                        torch.save(model, os.path.join(out_dir,best_rmse['name']))
                re = relative_error(y_test=Y_val.flatten(), y_predict=y_[-1], threshold=0.7)
                print('epoch:{:<2d} | loss:{:<6.7f} | MAPE:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}'.format(epoch, loss, mape, rmse, re))

    #         if (len(loss_list) > 1) and (abs(loss_list[-2] - loss_list[-1]) < 1e-9):
    #           break
        mape, rmse = evaluation(y_test=denormalize(Y_val).flatten(), y_predict=denormalize(np.array(y_[-1])))
        re = relative_error(y_test=Y_val.flatten(), y_predict=y_[-1], threshold=0.7)
        return model, y_[-1], mape_list, rmse_list, re

    def test(self, model, X_test, Y_test, out_dir, points_near=0, starting_point=0, device='cpu', pin=None):
        X_local = X_test
        X_local = np.array(X_local).astype(np.float32)
        Y = np.array(Y_test).astype(np.float32)
        X_local = torch.from_numpy(X_local).to(device)
        X_local = torch.reshape(X_local, (X_local.size(0),1,-1))
        Y = torch.from_numpy(Y).to(device)
        peak_idx = peak_visualization(Y_test=Y_test,points_near=points_near,pin=pin)
        model.to(device)

        model.eval()
        with torch.no_grad():
            Y_pred, _ = model(X_local)
            pred = denormalize(Y_pred[starting_point:])

        test = denormalize(Y_test[starting_point:])

        plt.figure(figsize=(7,5))
        if starting_point == 0:
            plt.plot(pred)
            plt.plot(test)
            plt.legend(['Predict', 'Test'])
        else:
            plt.plot(np.concatenate((denormalize(Y_test[-60:starting_point]), pred)))
            plt.plot(denormalize(Y_test[-60:]))
            plt.legend(['Predict', 'Test'])
        # plt.plot(peak_idx, test[peak_idx], ".", color='brown')
        plt.savefig(out_dir + '/predict.png', format='png')

        peak_mape = mean_absolute_percentage_error(test[peak_idx], pred[peak_idx])
        peak_rmse = root_mean_square_error(test[peak_idx], pred[peak_idx])

        non_peak_idx = list(set(range(len(test))) - set(peak_idx))

        non_peak_mape = mean_absolute_percentage_error(test[non_peak_idx], pred[non_peak_idx])
        non_peak_rmse = root_mean_square_error(test[non_peak_idx], pred[non_peak_idx])

        mape = mean_absolute_percentage_error(test, pred)
        rmse = root_mean_square_error(test, pred)

        with open(out_dir + '/result.txt', 'a') as f:
            f.write("overall rmse:" + str(rmse) + '\n')
            f.write("overall mape:" + str(mape) + '\n')
            f.write("peak rmse:" + str(peak_rmse) + '\n')
            f.write("peak mape:" + str(peak_mape) + '\n')
            f.write("non peak rmse:" + str(non_peak_rmse) + '\n')
            f.write("non peak mape:" + str(non_peak_mape) + '\n')
        print("Predict successfully. Check prediction figure at " + out_dir + '/predict.png')

        # return rmse, mape, peak_rmse, peak_mape, non_peak_rmse, non_peak_mape
        return pred, test