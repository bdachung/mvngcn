import numpy as np
import random
import math
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import sqrt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from utils import setup_seed, evaluation, relative_error, denormalize, peak_visualization, mean_absolute_percentage_error,root_mean_square_error, print_num_params
import copy
import functools
from sklearn.model_selection import train_test_split
import time
from torch.utils.data import TensorDataset
from tqdm import tqdm

setup_seed(0)

def masked_softmax(X, valid_lens, using_gumble=False, tau=1, hard=False, eps=1e-10, dim=-1):
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        if using_gumble:
            return F.gumbel_softmax(logits=X, tau=tau, hard=hard, eps=eps, dim=dim)
        else:
            return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        if using_gumble:
            return F.gumbel_softmax(logits=X.reshape(X), tau=tau, hard=hard, eps=eps, dim=dim)
        else:
            return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout, num_heads=None, using_gumble=False, tau=1, hard=False, eps=1e-10, dim=-1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads  # To be covered later
        self.using_gumble = using_gumble
        self.tau = tau
        self.hard = False
        self.eps = eps
        self.dim = dim

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None,
                window_mask=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        if window_mask is not None:  # To be covered later
            num_windows = window_mask.shape[0]
            n, num_queries, num_kv_pairs = scores.shape
            # Shape of window_mask: (num_windows, no. of queries,
            # no. of key-value pairs)
            scores = scores.reshape((n // (num_windows * self.num_heads), num_windows, self.
    num_heads, num_queries, num_kv_pairs)) + window_mask.unsqueeze(1).unsqueeze(0)
            scores = scores.reshape((n, num_queries, num_kv_pairs))
        self.attention_weights = masked_softmax(scores, valid_lens, using_gumble=self.using_gumble, tau=self.tau, hard=self.hard, eps=self.eps, dim=self.dim)
        return torch.bmm(self.dropout(self.attention_weights), values), self.attention_weights

class DilatedCNN(nn.Module):
    def __init__(self, in_channels, out_channels, pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation):
        super().__init__()

        self.pad = pcnn_padding
        self.conv1d = nn.Conv1d(in_channels, out_channels, pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation)

    def forward(self, X):
        X = self.conv1d(X)
        return X[:,:,:-self.pad]

from torch.nn import Linear
from torch_geometric.nn import GCNConv

def gnn_data(input, adjacency_matrix):
    data_list = []
    for i in range(len(input)):
        row, col = np.where(adjacency_matrix != 0)
        coo = np.array(list(zip(row,col)))
        coo = np.reshape(coo, (2,-1))
        data = Data(x=torch.tensor(input[i], dtype=torch.float),edge_index=torch.tensor(coo, dtype=torch.long))
        data_list.append(data)
    return DataLoader(data_list,input.shape[0])

class TabularGraph(nn.Module):
    # def __init__(self, local_num_features, local_out_size, global_num_features, global_out_size, window_size, kernel, stride, padding, dilation, hidden_size, output_size, dropout=0, using_gumble=False, aggr='mean'):
    def __init__(self, num_features, window_size, kernel, stride, padding, dilation, hidden_size, output_size, dropout=0, using_gumble=False, aggr='mean', threshold=0.05, tau=1, hard=False, eps=1e-10, dim=-1):
        super().__init__()
        self.i = 0
        self.threshold = threshold

        self.dccnn = DilatedCNN(num_features, num_features, kernel, stride, padding, dilation)

        self.Wq_time = torch.nn.Parameter(torch.randn(window_size, window_size))
        self.Wk_time = torch.nn.Parameter(torch.randn(window_size, window_size))
        self.Wv_time = torch.nn.Parameter(torch.randn(window_size, window_size))

        self.Wq_feat = torch.nn.Parameter(torch.randn(num_features, num_features))
        self.Wk_feat = torch.nn.Parameter(torch.randn(num_features, num_features))
        self.Wv_feat = torch.nn.Parameter(torch.randn(num_features, num_features))

        self.attention = DotProductAttention(dropout, using_gumble=using_gumble, tau=tau, hard=hard, eps=eps, dim=dim)
    
        self.gcn_time1 = GCNConv(num_features, hidden_size)
        self.gcn_time2 = GCNConv(hidden_size, hidden_size)
        self.gcn_time3 = GCNConv(hidden_size, hidden_size)
        self.gcn_feat1 = GCNConv(window_size, hidden_size)
        self.gcn_feat2 = GCNConv(hidden_size, hidden_size)
        self.gcn_feat3 = GCNConv(hidden_size, hidden_size)

        self.relu = nn.ReLU()

        if aggr == 'mean':
            self.pool = torch.mean
        elif aggr == 'max':
            self.pool = torch.max
        elif aggr == 'add':
            self.pool = torch.sum

        self.proj = Linear(hidden_size*2, output_size)
        self.linear_branch = Linear(window_size, output_size)
        self.final1 = Linear(2*output_size, 16)
        self.final2 = Linear(16, 8)
        self.final3 = Linear(8, 1)

    def forward(self, X_time):
        linear_capacity = self.linear_branch(torch.flatten(X_time[:,:,-1], start_dim=1))

        if X_time.get_device() == -1:
            device = 'cpu'
        else:
            device = 'cuda'
        # batch_size, num_features, window_size
        X_feat = torch.reshape(X_time, (X_time.size(0), X_time.size(2), X_time.size(1)))
        X_feat = self.dccnn(X_feat)

        Q_feat = torch.einsum('jj,ijk->ijk', self.Wq_feat, X_feat)
        K_feat = torch.einsum('jj,ijk->ijk', self.Wk_feat, X_feat)
        # K_feat = torch.einsum('jj,ijk->ijk', self.Wq_feat, X_feat)
        V_feat = torch.einsum('jj,ijk->ijk', self.Wv_feat, X_feat)

        _, weight_feat = self.attention(Q_feat, K_feat, V_feat)

        adjacency_matrix = torch.ones((X_feat.size(1), X_feat.size(1))).long().to(device)
        edge_index = adjacency_matrix.nonzero().t().contiguous()
        weight_feat = torch.mean(weight_feat, dim=0)
        weight_feat = weight_feat.flatten()
        out_feat = self.gcn_feat1(X_feat, edge_index, weight_feat)
        # out_feat = self.relu(out_feat)
        out_feat = self.gcn_feat2(out_feat, edge_index, weight_feat)
        out_feat = self.gcn_feat3(out_feat, edge_index, weight_feat)
        # out_feat = self.relu(out_feat)
        # 1, hidden_size
        out_feat = self.pool(out_feat, dim=1)

        Q_time = torch.einsum('jj,ijk->ijk', self.Wq_time, X_time)
        K_time = torch.einsum('jj,ijk->ijk', self.Wk_time, X_time)
        # K_time = torch.einsum('jj,ijk->ijk', self.Wq_time, X_time)
        V_time = torch.einsum('jj,ijk->ijk', self.Wv_time, X_time)

        _, weight_time = self.attention(Q_time, K_time, V_time)

        adjacency_matrix = torch.ones((X_time.size(1), X_time.size(1))).long().to(device)
        edge_index = adjacency_matrix.nonzero().t().contiguous()
        weight_time = torch.mean(weight_time, dim=0)
        weight_time = weight_time.flatten()
        out_time = self.gcn_time1(X_time, edge_index, weight_time)
        # out_time = self.relu(out_time)
        out_time = self.gcn_time2(out_time, edge_index, weight_time)
        out_time = self.gcn_time3(out_time, edge_index, weight_time)
        # out_time = self.relu(out_time)

        # 1, hidden_size
        out_time = self.pool(out_time, dim=1)

        # print(weights)
        out = torch.concat([out_feat, out_time], dim=-1)

        nonlinear_capacity = self.proj(out)

        out = torch.concat([nonlinear_capacity, linear_capacity], dim=-1)

        out = self.final3(self.final2(self.final1(out)))

        return out, [weight_feat, weight_time]


def extract_batter_features(concat_df, global_df, battery_nb, window_size=8, prediction_interval=1, extract_points_len=11, multichannel_features=['voltage','current','resistance'], global_seq_features = ['x5', 'rest_period', 'capacity'], key='cycle', label='capacity'):
    # choose only 1 battery at a time
    #   df_copy = concat_df[battery_nb].copy()
    df_copy = concat_df[battery_nb].copy()
    # rest time of each charge cycle
    #   global_df_copy = global_df[battery_nb].copy()
    global_df_copy = global_df[battery_nb].copy()
        
    capacity = df_copy.groupby([key]).mean()[label].to_numpy().reshape(-1, 1)

    global_feature = global_df_copy[global_seq_features].to_numpy()

    # sliding window
    def sliding_window(window_length, input_matrix):
        nb_items = input_matrix.shape[0]
        sub_window = np.array([np.arange(start=x, stop=x+window_length, step=1) for x in range(nb_items-window_length)])
        return input_matrix[sub_window]

    slided_glb_feature = sliding_window(window_size, global_feature)
    #   print(slided_mc_feature.shape)
    slided_capacity = capacity[window_size:]
    #   print(slided_capacity.shape)

    # shifting forecast interval
    input = slided_glb_feature[:-prediction_interval]
    output = slided_capacity[prediction_interval:]

    return input, output

def cross_validation(batteries, df, rest_df, extract_points_len=11, multichannel_features=['voltage','current','resistance'], window_size=8, prediction_interval=1, global_seq_features = ['x5', 'rest_period', 'capacity'], key='cycle', label='capacity', validation=None, random_state=7, shuffle=False):
        # last pin is the test set
        all_batteries = [extract_batter_features(df, rest_df, battery_nb, window_size=window_size, prediction_interval=prediction_interval, extract_points_len=extract_points_len, multichannel_features=multichannel_features, global_seq_features=global_seq_features, key=key, label=label) for battery_nb in batteries]

        # cross validation
        train_set = all_batteries[0:-1]
        val_set = all_batteries[-1]
        test_set = all_batteries[-1]

        glb_train = functools.reduce(lambda a,b: np.concatenate((a,b), axis=0), [x for x,_ in train_set]).reshape(-1, window_size, len(global_seq_features))

        Y_train = functools.reduce(lambda a,b: np.concatenate((a,b), axis=0), [y for _,y in train_set])
        Y_train = Y_train.reshape(-1, 1)

        glb_test, Y_test = test_set
        X_test = glb_test

        if validation is None:
            X_train = glb_train

            glb_val, Y_val = val_set
            X_val = glb_val

        else:
            glb_train, glb_val, Y_train, Y_val = train_test_split(glb_train, Y_train, test_size=validation, random_state=random_state, shuffle=shuffle)
            X_train = glb_train
            X_val = glb_val

        return X_train, Y_train, X_val, Y_val, X_test,Y_test

def train(model, X_train, Y_train, X_val, Y_val, out_dir, batch_size=32, EPOCH=1000, LR=0.001, seed=0, weight_decay=0., device="cpu", clipping_value=None, verbose=2, patience=25):
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        print("Start training")
        
        loss_list, y_ = [0], [] 
        rmse_list, mape_list = [], []
        best_rmse = None

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        train_ds = TensorDataset(torch.from_numpy(X_train).float().to(device), torch.from_numpy(Y_train).float().to(device))
        train_dl = DataLoader(train_ds, batch_size=batch_size)

        val_ds = TensorDataset(torch.from_numpy(X_val).float().to(device), torch.from_numpy(Y_val).float().to(device))
        val_dl = DataLoader(val_ds, batch_size=batch_size)

        R_avg = []
        for epoch in tqdm(range(EPOCH),desc='train'):
            model.train()
            _loss = []
            for X, Y in train_dl:
                output, attention_weights = model(X)
                # print(output.shape)
                downstream_loss = criterion(output, Y)
                loss = downstream_loss
                _loss.append(loss.detach().cpu().numpy())

                optimizer.zero_grad()
                loss.backward()
                if clipping_value:
                    nn.utils.clip_grad_norm(model.parameters(), clipping_value)
                optimizer.step()

            if epoch%verbose==0 or epoch==(EPOCH-1):
                point_list = np.array([])
                for x, y in val_dl:
                    model.eval()
                    with torch.no_grad():
                        pred, _ = model(x)
                    point_list = np.append(point_list, (pred[:,0].cpu().detach().numpy()))
                y_.append(point_list)
                loss_list.append(np.mean(_loss))
                mape, rmse = evaluation(y_test=denormalize(Y_val).flatten(), y_predict=denormalize(np.array(y_[-1].flatten())))
                mape_list.append(mape)
                rmse_list.append(rmse)
                if best_rmse == None:
                    best_rmse = {}
                    best_rmse['name'] = 'checkpoint-' + str(epoch) + '.pt'
                    best_rmse['value'] = rmse
                    best_rmse['model'] = copy.deepcopy(model)
                    best_rmse['epoch'] = epoch
                else:
                    if rmse < best_rmse['value']:
                        best_rmse['name'] = 'checkpoint-' + str(epoch) + '.pt'
                        best_rmse['value'] = rmse
                        del best_rmse['model']
                        best_rmse['model'] = copy.deepcopy(model)
                        best_rmse['epoch'] = epoch
                re = relative_error(y_test=Y_val.flatten(), y_predict=y_[-1], threshold=0.7)
                print('epoch:{:<2d} | loss:{:<6.7f} | MAPE:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}'.format(epoch, loss, mape, rmse, re))
                if (epoch - best_rmse['epoch']) / verbose >= patience:
                    print("Early stopping")
                    break
                
            if (len(loss_list) > 1) and (abs(loss_list[-2] - loss_list[-1]) < 1e-9):
                print("loss convergence")
                break

        torch.save(best_rmse['model'], os.path.join(out_dir,'checkpoint.pt'))
        mape, rmse = evaluation(y_test=denormalize(Y_val).flatten(), y_predict=denormalize(np.array(y_[-1])))
        re = relative_error(y_test=Y_val.flatten(), y_predict=y_[-1], threshold=0.7)

        print("Train successfully")

        return model, y_[-1], mape_list, rmse_list, re

def test(model, X_test, Y_test, out_dir, points_near=0, device='cpu', starting_point=0, pin=None):
    test_ds = TensorDataset(torch.from_numpy(X_test).float().to(device), torch.from_numpy(Y_test).float().to(device))
    test_dl = DataLoader(test_ds, batch_size=1)
    peak_idx = peak_visualization(Y_test=Y_test,points_near=points_near,pin=pin)
    model.to(device)
    model.eval()
    Y_pred = np.array([])
    with torch.no_grad():
        for x, _ in test_dl:
            _y_pred, _ = model(x)
            Y_pred = np.append(Y_pred, (_y_pred[:,0].cpu().detach().numpy()))

    pred = denormalize(Y_pred.flatten()[starting_point:])
    test = denormalize(Y_test.flatten()[starting_point:])

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
    return pred, test

if __name__ == '__main__':
    Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
    Battery = np.load('./calcev4/concat_df.npy', allow_pickle=True)
    Battery = Battery.item()

    rest_time = np.load('./calcev4/global_df.npy', allow_pickle=True)
    rest_time = rest_time.item()

    window_size=8
    prediction_interval=1
    multichannel_features = ['voltage','current','resistance']
    global_seq_features = ['x5', 'rest_period', 'capacity']
    epochs = EPOCH = 1000
    BATTERIES = ['CS2_38','CS2_35', 'CS2_36','CS2_37']
    extract_points_len = 11
    lr = 0.001
    kernel_size = 5
    batch_size = 8
    hidden_size = 16
    output_size = 8
    aggr = 'mean'

    from datetime import datetime
    t = datetime.now()
    t_str = t.strftime("%d_%m_%Y_%H_%M")

    if not os.path.isdir(f'{t_str}_time_feat_Node'):
      os.mkdir(f'{t_str}_time_feat_Node')
    
    f = open(f'{t_str}_time_feat_Node/config.txt', 'a')
    f.write(f'global features: {global_seq_features}\n')
    f.write(f'epochs: {epochs}\n')
    f.write(f'lr: {lr}\n')
    f.write(f'kernel_size: {kernel_size}\n')
    f.write(f'batch_size: {batch_size}\n')
    f.write(f'aggr: {aggr}\n')
    f.write(f'hidden_size: {hidden_size}\n')
    f.write(f'output_size: {output_size}\n')
    f.close()

    for bat in BATTERIES:
        batteries = BATTERIES.copy()
        batteries.remove(bat)
        batteries = batteries + [bat]

        X_train, Y_train, X_val, Y_val, X_test,Y_test = cross_validation(batteries, Battery, rest_time, extract_points_len=extract_points_len, multichannel_features=multichannel_features, window_size=window_size, prediction_interval=prediction_interval, global_seq_features = global_seq_features, key='cycle', label='capacity', random_state=7, shuffle=True, validation=0.1)

        model = TabularGraph(len(global_seq_features), window_size, kernel_size, 1, kernel_size-1, 1, hidden_size, output_size, 0, False, aggr=aggr, threshold=0.001)

        print_num_params(model)

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'
        start = time.time()
        train(model, X_train, Y_train, X_val, Y_val, out_dir=f'{t_str}_time_feat_Node/' + str(batteries[-1]),EPOCH=epochs,LR=lr,verbose=10, device=device,patience=30, batch_size=batch_size)
        end = time.time()

        with open(f'{t_str}_time_feat_Node/' + str(batteries[-1]) + '/result.txt', 'a') as f:
            f.write(f"Training time {end - start} seconds" +'\n')
        print(f"Training time {end - start} seconds")

        model = torch.load(f'{t_str}_time_feat_Node/' + str(batteries[-1]) + '/checkpoint.pt')

        test(model, X_test, Y_test, out_dir=f'{t_str}_time_feat_Node/' + str(batteries[-1]), pin=str(batteries[-1]), device=device, points_near=1)