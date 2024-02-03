import numpy as np
import random
import math
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.signal import find_peaks

print('hi')


import matplotlib.pyplot as plt

Rated_Capacity = 1.1

def denormalize(data, rated_capacity=Rated_Capacity):
    return data * (2.035337591005598 - 1.15381833159625) + 1.15381833159625

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def root_mean_square_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return math.sqrt(np.mean((y_true - y_pred)**2))

def drop_outlier(array,count,bins):
    index = []
    range_ = np.arange(1,count,bins)
    for i in range_[:-1]:
        array_lim = array[i:i+bins]
        sigma = np.std(array_lim)
        mean = np.mean(array_lim)
        th_max,th_min = mean + sigma*2, mean - sigma*2
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        idx = idx[0] + i
        index.extend(list(idx))
    return np.array(index)


def build_sequences(text, window_size):
    #text:list of capacity
    x, y = [],[]
    for i in range(len(text) - window_size):
        sequence = text[i:i+window_size]
        target = text[i+1:i+1+window_size]

        x.append(sequence)
        y.append(target)

    return np.array(x), np.array(y)


##Hung##
#Change: add parameter "feature_names"
def get_train_test(data_dict, name, feature_names, window_size=8, ):
    data_sequence=data_dict[name][feature_names]
    train_data, test_data = data_sequence[:window_size+1], data_sequence[window_size+1:]
    train_x, train_y = build_sequences(text=train_data, window_size=window_size)
    for k, v in data_dict.items():
        if k != name:
            data_x, data_y = build_sequences(text=v[feature_names], window_size=window_size)
            train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]
            
    return train_x, train_y, list(train_data), list(test_data)


def relative_error(y_test, y_predict, threshold):
    true_re, pred_re = len(y_test), 0
    for i in range(len(y_test)-1):
        if y_test[i] <= threshold >= y_test[i+1]:
            true_re = i - 1
            break
    for i in range(len(y_predict)-1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
    return abs(true_re - pred_re)/true_re if abs(true_re - pred_re)/true_re<=1 else 1


def evaluation(y_test, y_predict):
    mape = mean_absolute_percentage_error(y_test, y_predict)
    rmse = root_mean_square_error(y_test, y_predict)
    return mape, rmse
    
    
def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(seed) # 为CPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def peak_visualization(Y_test, points_near=2, pin=None):
    Y_test = denormalize(Y_test)
    peak_idx = np.array(find_peaks(Y_test.reshape(-1, Y_test.shape[0])[0], height = 0.0001)[0])

    # fig, ax = plt.subplots(1, figsize=(12, 8))
    # ax.plot(Y_test, label='Actual Capacity')
    # ax.plot(peak_idx, Y_test[peak_idx], ".", color='brown')
    # print(f'Number of peak of Battery {pin} : {len(peak_idx)}')
    # ax.set(xlabel='Discharge cycles', ylabel='Capacity/Ah', title='Peak detection of Battery ' + pin)
    # plt.legend()

    peak_len = len(peak_idx)
    if points_near != 0:
        peak_idx = np.append(peak_idx, [peak_idx[-1] + i for i in range(1,points_near+1)])
        peak_idx = np.insert(peak_idx, peak_len - 1, [peak_idx[-1] - i for i in range(points_near,0,-1)])
        for idx in range(peak_len - 2, -1, -1):
            peak_idx = np.insert(peak_idx, idx + 1, [peak_idx[idx] + i for i in range(1,points_near+1)])
            peak_idx = np.insert(peak_idx, idx, [peak_idx[idx] - i for i in range(points_near,0,-1)])

    peak_idx = np.array(sorted(set(peak_idx)))
    peak_idx = np.intersect1d(peak_idx, range(len(Y_test)))

    # fig, ax = plt.subplots(1, figsize=(12, 8))
    # ax.plot(Y_test, label='Actual Capacity')
    # ax.plot(peak_idx, Y_test[peak_idx], ".", color='brown')
    # print(f'Number of peak of Battery {pin} : {len(peak_idx)}')
    # ax.set(xlabel='Discharge cycles', ylabel='Capacity/Ah', title='Group of peak detection of Battery ' + pin)
    # plt.legend()
    return peak_idx

def error_computation(model, X_test, Y_test, peak_idx):
    Y_pred = model.predict(X_test)
    Y_pred = denormalize(Y_pred)
    Y_test = denormalize(Y_test)

    peak_mape = mean_absolute_percentage_error(Y_test[peak_idx], Y_pred[peak_idx])
    peak_rmse = root_mean_square_error(Y_test[peak_idx], Y_pred[peak_idx])
    print(f'\tPeak Error: \nMAPE: {mape}\nRMSE: {rmse}')

    non_peak_idx = list(set(range(len(Y_test))) - set(peak_idx))

    non_peak_mape = mean_absolute_percentage_error(Y_test[non_peak_idx], Y_pred[non_peak_idx])
    non_peak_rmse = root_mean_square_error(Y_test[non_peak_idx], Y_pred[non_peak_idx])
    print(f'\tNon-Peak Error: \nMAPE: {mape}\nRMSE: {rmse}')

    mape = mean_absolute_percentage_error(Y_test, Y_pred)
    rmse = root_mean_square_error(Y_test, Y_pred)
    print(f'\tOverall Error: \nMAPE: {mape}\nRMSE: {rmse}')
    return rmse, mape, peak_rmse, peak_mape, non_peak_rmse, non_peak_mape

def print_num_params(model):
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


import copy
import functools
from sklearn.model_selection import train_test_split
import time
from torch.utils.data import TensorDataset
from tqdm import tqdm
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader as geometric_DataLoader
from torch_geometric.data import Data




def gnn_data(input, edge_index, edge_weight):
    assert len(input.shape) == 3
    assert len(edge_index.shape) == 2
    assert len(edge_weight.shape) == 2

    class GraphDataset(Dataset):
      def __init__(self, graphs):
        self._graphs = graphs
      def __len__(self):
        return len(self._graphs)
      def __getitem__(self, idx):
        return self._graphs[idx]

    data = [Data(input[i], edge_index, edge_weight[i]) for i in range(len(input))]

    dataset = GraphDataset(data)

    return geometric_DataLoader(dataset, input.shape[0])

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

class AddNorm(nn.Module):
    """Residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        
        self.P[:, :, 0::2] = torch.sin(X[:,:num_hiddens//2 + num_hiddens%2])
        self.P[:, :, 1::2] = torch.cos(X[:,:num_hiddens//2])

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class TransformerDecoderBlock(nn.Module):
    @staticmethod
    def calc_cnn_output_shape(
        dim, ksize, stride=1, padding=0, dilation=1
    ):
        def shape_each_dim(i):
            odim_i = dim[i] + 2 * padding - dilation * (ksize - 1) - 1
            return int(odim_i / stride) + 1
        return [shape_each_dim(0), shape_each_dim(1)]

    def __init__(self, idim, dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation, dropout, 
                pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation, using_gumble=False):
        super().__init__()
        H, W = idim[0], idim[1]

        self.dccnn1 = DilatedCNN(H, H, dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation)
#         lnshape = TransformerDecoderBlock.calc_cnn_output_shape((H,W), dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation)
        self.dccnn_ln1 = AddNorm([H, W], dropout)
        
        self.dccnn2 = DilatedCNN(H, H, dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation)
#         lnshape = TransformerDecoderBlock.calc_cnn_output_shape(lnshape, dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation)
        self.dccnn_ln2 = AddNorm([H, W], dropout)

        self.Wq = torch.nn.Parameter(torch.randn(H, H))
        self.Wk = torch.nn.Parameter(torch.randn(H, H))
        self.Wv = torch.nn.Parameter(torch.randn(H, H))
        # self.Wq = nn.Linear(lnshape[1],lnshape[1])
        # self.Wk = nn.Linear(lnshape[1],lnshape[1])
        # self.Wv = nn.Linear(lnshape[1],lnshape[1])

        self.attention = DotProductAttention(dropout,using_gumble=using_gumble)
        self.attention_weight = None
        self.attention_ln = AddNorm((H, W), dropout)
        
        self.pcnn1 = nn.Conv1d(H, H, pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation)
#         lnshape = TransformerDecoderBlock.calc_cnn_output_shape(lnshape, pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation)
        # self.pcnn_ln1 = AddNorm([*lnshape], dropout)
        
        self.pcnn2 = nn.Conv1d(H, H, pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation)
#         lnshape = TransformerDecoderBlock.calc_cnn_output_shape(lnshape, pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation)
        # self.pcnn_ln2 = AddNorm([*lnshape], dropout)
    
#     def forward(self, X, Y):
    def forward(self, X):
        #input (B, H, W) 
        X = self.dccnn_ln1(X,self.dccnn1(X))
        X = self.dccnn_ln2(X,self.dccnn2(X))

        
        Q = torch.einsum('jj,ijk->ijk', self.Wq, X)
        K = torch.einsum('jj,ijk->ijk', self.Wk, X)
        V = torch.einsum('jj,ijk->ijk', self.Wv, X)
        # Q = self.Wq(X)
        # K = self.Wq(X)
        # V = self.Wv(X)

        Y, self.attention_weight = self.attention(Q,K,V)
        # Y = self.attention(Q,K,V)

        X = self.attention_ln(X, Y)
        X = self.pcnn1(X)
        X = self.pcnn2(X)
        return X, self.attention_weight

class Decoder(nn.Module):
    def __init__(self, idim, dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation, dropout, 
                pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation, n_layers, using_gumble=False):
        super().__init__()
        
        self.layers = nn.ModuleList([TransformerDecoderBlock(
            idim, 
            dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation, 
            dropout, 
            pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation, using_gumble=using_gumble
        ) for _ in range(n_layers)])
        self.attention_weights = []
#     def forward(self, X, Y):
    def forward(self, X):
        self.attention_weights = []
        for layer in self.layers:
#             X = layer(X, Y)
            X, attention_weight = layer(X)
            # X = layer(X)
            self.attention_weights.append(attention_weight)
        
        return X, self.attention_weights
        # return X

class Post_Decoder(nn.Module):
    def __init__(self, num_var=None, in_dim=1, out_dim=1):
        super().__init__()
        self.num_var = num_var
        self.proj = nn.Linear(in_dim, out_dim)
        if num_var:
            self.linear = nn.Linear(num_var*out_dim,1)
        else:
            self.linear = nn.Linear(out_dim,1)
    def forward(self, X):
        X = self.proj(X)
        if self.num_var:
            X = X.view(X.size(0),-1)
        return self.linear(X)

class TimeTermGCN(nn.Module):
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
        # self.gcn_time2 = GCNConv(hidden_size, hidden_size)
        # self.gcn_time3 = GCNConv(hidden_size, hidden_size)
        self.gcn_feat1 = GCNConv(window_size, hidden_size)
        # self.gcn_feat2 = GCNConv(hidden_size, hidden_size)
        # self.gcn_feat3 = GCNConv(hidden_size, hidden_size)

        self.relu = nn.ReLU()

        if aggr == 'mean':
            self.pool = torch.mean
        elif aggr == 'max':
            self.pool = torch.max
        elif aggr == 'add':
            self.pool = torch.sum

        self.proj = Linear(hidden_size*2, output_size)

    def forward(self, X_time):

        if X_time.get_device() == -1:
            device = 'cpu'
        else:
            device = 'cuda'
        # batch_size, num_features, window_size
        X_feat = torch.reshape(X_time, (X_time.size(0), X_time.size(2), X_time.size(1)))
        X_feat = self.dccnn(X_feat)

        Q_feat = torch.einsum('jj,ijk->ijk', self.Wq_feat, X_feat)
        K_feat = torch.einsum('jj,ijk->ijk', self.Wk_feat, X_feat)
        V_feat = torch.einsum('jj,ijk->ijk', self.Wv_feat, X_feat)

        _, weight_feat = self.attention(Q_feat, K_feat, V_feat)

        adjacency_matrix = torch.ones((X_feat.size(1), X_feat.size(1))).long().to(device)
        edge_index = adjacency_matrix.nonzero().t().contiguous()
        # weight_feat = torch.mean(weight_feat, dim=0)
        # weight_feat = weight_feat.flatten()
        weight_feat = weight_feat.view(weight_feat.size(0), -1)
        data_feat = gnn_data(X_feat, edge_index, weight_feat)
        _, batch_feat = next(enumerate(data_feat))
        out_feat = self.gcn_feat1(batch_feat.x, batch_feat.edge_index, batch_feat.edge_weight)
        out_feat = global_mean_pool(out_feat, batch_feat.batch)
        # out_feat = self.gcn_feat1(X_feat, edge_index, weight_feat)
        # out_feat = self.gcn_feat2(out_feat, edge_index, weight_feat)
        # out_feat = self.gcn_feat3(out_feat, edge_index, weight_feat)
        # 1, hidden_size
        # out_feat = self.pool(out_feat, dim=1)


        Q_time = torch.einsum('jj,ijk->ijk', self.Wq_time, X_time)
        K_time = torch.einsum('jj,ijk->ijk', self.Wk_time, X_time)
        V_time = torch.einsum('jj,ijk->ijk', self.Wv_time, X_time)

        _, weight_time = self.attention(Q_time, K_time, V_time)

        adjacency_matrix = torch.ones((X_time.size(1), X_time.size(1))).long().to(device)
        edge_index = adjacency_matrix.nonzero().t().contiguous()
        # weight_time = torch.mean(weight_time, dim=0)
        # weight_time = weight_time.flatten()
        weight_time = weight_time.view(weight_time.size(0), -1)
        data_time = gnn_data(X_time, edge_index, weight_time)
        _, batch_time = next(enumerate(data_time))
        out_time = self.gcn_time1(batch_time.x, batch_time.edge_index, batch_time.edge_weight)
        out_time = global_mean_pool(out_time, batch_time.batch)

        # out_time = self.gcn_time1(X_time, edge_index, weight_time)
        # out_time = self.gcn_time2(out_time, edge_index, weight_time)
        # out_time = self.gcn_time3(out_time, edge_index, weight_time)
        # 1, hidden_size
        # out_time = self.pool(out_time, dim=1)

        # print(weights)
        out = torch.concat([out_feat, out_time], dim=-1)

        # batch x 1 x output_size
        out = self.proj(out)

        # batch x output_size
        out = torch.reshape(out, (out.size(0), -1))

        return out, [weight_feat, weight_time]

class LocalCTARNS_TimeTermGCN(nn.Module):
    def __init__(self, local_feature_size, num_extracted_point, window_size, global_feature_size, frt_feature_size, output_node=1, dccnn_kernel_size=5, dccnn_stride=1, dccnn_padding=4, dccnn_dilation=1, dropout=0.01,  pcnn_kernel_size=1, pcnn_stride=1, pcnn_padding=0, pcnn_dilation=1, local_n_layers=1, global_n_layers=1, local_using_gb=False, global_using_gb=False, in_channels=1, hidden_channels=16, out_channels=16, K=2):
        super(LocalCTARNS_TimeTermGCN, self).__init__()
        self.local_ctarns_list = nn.ModuleList()
        self.local_postdecoder_list = nn.ModuleList()
        
        for _ in range(window_size):
            decoder = Decoder(
                (local_feature_size, num_extracted_point), 
                dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation, 
                dropout, 
                pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation,
                local_n_layers, using_gumble=local_using_gb)

            post_decoder = Post_Decoder(None, num_extracted_point, 1)
            
            self.local_ctarns_list.append(decoder)
            self.local_postdecoder_list.append(post_decoder)
        
        self.global_timetermgcn = TimeTermGCN(local_feature_size + global_feature_size, window_size, dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation, hidden_channels, out_channels, dropout)
        
        self.global_linear_layer = nn.Linear(global_feature_size, output_node)
        
        self.dense1 = nn.Linear(out_channels + window_size*output_node + frt_feature_size, 30)
        self.tanh = nn.Tanh()
        self.dense2 = nn.Linear(30, 10)
        self.dense3 = nn.Linear(10, 1)
        
    def forward(self, X_local, X_global, X_frt):
        # X_local.shape = (batch x cycle x x num_extracted_points x local_feature_size)
        # X_global.shape = (batch x cycle x global_feature_size)

        # batch x cycle x local_feature_size x num_extracted_points
        X_local = torch.reshape(X_local, (X_local.size(0),X_local.size(1),X_local.size(3),X_local.size(2)))
        Y = []
        for i in range(len(self.local_ctarns_list)):
            # batch x local_feature_size x 1
            y, attention_weights = self.local_ctarns_list[i](X_local[:,i,:,:])
            y = self.local_postdecoder_list[i](y)
            Y.append(y)
        # batch x local_feature_size x cycle
        Y = torch.cat(Y,axis=-1)

        # batch x cycle x output_node
        global_linear = self.global_linear_layer(X_global)

        # batch x cycle*output_node
        global_linear = global_linear.view(global_linear.size(0), -1)

        # batch x (local_feature_size + global_feature_size) x cycle
        Y = torch.cat((Y, torch.reshape(X_global, (-1,X_global.size(-1),X_global.size(-2)))),axis=1)

        # batch x cycle x (local_feature_size + global_feature_size)
        Y = torch.permute(Y, (0, 2, 1))
        
        # batch x out_channels
        Y, _ = self.global_timetermgcn(Y)

        # batch x (out_channels + cycle*output_node + frt_feature_size)
        Y = torch.cat([Y, global_linear, X_frt],axis=-1)

        # batch x 1
        return self.dense3(self.tanh(self.dense2(self.tanh(self.dense1(Y))))), []


def extract_batter_features(df, global_df, battery_name, window_size=8, prediction_interval=1, extract_points_len=11, multichannel_features=['voltage','current','resistance'], global_seq_features = ['x5', 'rest_period', 'capacity'], key='cycle', label='capacity', rest_period='rest_period'):
    """df: local features, global_df: global features"""
    # choose only 1 battery at a time
    df_copy = df[battery_name].copy()
    # last data point of every charge phase

    global_df_copy = global_df[battery_name].copy()

    # capacity = df_copy.groupby([key]).mean()[label].to_numpy().reshape(-1, 1)

    capacity = global_df_copy[label].to_numpy().reshape(-1, 1)

    global_feature = global_df_copy[global_seq_features].to_numpy()

    # capacity = df_copy.groupby([key]).mean()[label].to_numpy().reshape(-1, 1)
    rest_time = global_df_copy[rest_period].to_numpy().reshape(-1, 1)

    multi_channel_feature = df_copy[multichannel_features].to_numpy().reshape(-1, extract_points_len, len(multichannel_features))

    # sliding window
    def sliding_window(window_length, input_matrix):
        nb_items = input_matrix.shape[0]
        sub_window = np.array([np.arange(start=x, stop=x+window_length, step=1) for x in range(nb_items-window_length)])
        return input_matrix[sub_window]

    slided_mc_feature = sliding_window(window_size, multi_channel_feature)
    slided_glb_feature = sliding_window(window_size, global_feature)
    slided_forecast_rest_time = rest_time[window_size:]
    slided_capcity = capacity[window_size:]

    # shifting forecast interval
    input = (slided_mc_feature[:-prediction_interval], slided_glb_feature[:-prediction_interval], slided_forecast_rest_time[prediction_interval:])
    output = slided_capcity[prediction_interval:]

    return input, output

def cross_validation(batteries, df, rest_df, extract_points_len=11, multichannel_features=['voltage','current','resistance'], window_size=8, prediction_interval=1, global_seq_features = ['x5', 'rest_period', 'capacity'], key='cycle', label='capacity', rest_period='rest_period', validation=None, random_state=7, shuffle=False):
    # last pin is the test set
    all_batteries = [extract_batter_features(df, rest_df, battery_nb, window_size=window_size, prediction_interval=prediction_interval, extract_points_len=extract_points_len, multichannel_features=multichannel_features, global_seq_features=global_seq_features, key=key, label=label, rest_period=rest_period) for battery_nb in batteries]

    # cross validation
    train_set = all_batteries[0:-1]
    val_set = all_batteries[-1]
    test_set = all_batteries[-1]

    mc_train = functools.reduce(lambda a,b: np.concatenate((a,b), axis=0), [x for (x, _, _),_ in train_set]).reshape(-1, window_size, extract_points_len, len(multichannel_features))
    glb_train = functools.reduce(lambda a,b: np.concatenate((a,b), axis=0), [x for (_, x, _),_ in train_set]).reshape(-1, window_size, len(global_seq_features))
    frt_train = functools.reduce(lambda a,b: np.concatenate((a,b), axis=0), [x for (_, _, x),_ in train_set]).reshape(-1, 1)

    Y_train = functools.reduce(lambda a,b: np.concatenate((a,b), axis=0), [y for _,y in train_set])
    Y_train = Y_train.reshape(-1, 1)

    (mc_test, glb_test, frt_test), Y_test = test_set
    X_test = [mc_test, glb_test, frt_test]

    if validation is None:
        X_train = [mc_train, glb_train, frt_train]

        (mc_val, glb_val, frt_val), Y_val = val_set
        X_val = [mc_val, glb_val, frt_val]

    else:
        mc_train, mc_val, glb_train, glb_val, frt_train, frt_val, Y_train, Y_val = train_test_split(mc_train, glb_train, frt_train, Y_train, test_size=validation, random_state=random_state, shuffle=shuffle)
        X_train = [mc_train, glb_train, frt_train]
        X_val = [mc_val, glb_val, frt_val]

    return X_train, Y_train, X_val, Y_val, X_test,Y_test

class NestedDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index], self.data[2][index], self.data[3][index]
    
    def __len__(self):
        return self.data[0].size(0)

def train(model, X_train, Y_train, X_val, Y_val, out_dir, batch_size=32, EPOCH=1000, LR=0.001, seed=0, weight_decay=0., device="cpu", clipping_value=None, verbose=2, patience=25):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    print("Start training")
    # setup_seed(seed)

    loss_list, y_ = [0], [] 
    rmse_list, mape_list = [], []
    best_rmse = None

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    criterion = nn.MSELoss()

#     train_ds = TensorDataset(torch.from_numpy(X_train).float().to(device), torch.from_numpy(Y_train).float().to(device))
#     train_dl = DataLoader(train_ds, batch_size=batch_size)
    
#     val_ds = TensorDataset(torch.from_numpy(X_val).float().to(device), torch.from_numpy(Y_val).float().to(device))
#     val_dl = DataLoader(val_ds, batch_size=batch_size)
    
    train_ds = NestedDataset([
        torch.from_numpy(X_train[0]).float().to(device),
        torch.from_numpy(X_train[1]).float().to(device),
        torch.from_numpy(X_train[2]).float().to(device),
        torch.from_numpy(Y_train).float().to(device)
    ])
    train_dl = DataLoader(train_ds, batch_size=batch_size)
    
    val_ds = NestedDataset([
        torch.from_numpy(X_val[0]).float().to(device),
        torch.from_numpy(X_val[1]).float().to(device),
        torch.from_numpy(X_val[2]).float().to(device),
        torch.from_numpy(Y_val).float().to(device)
    ])
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    R_avg = []
    for epoch in tqdm(range(EPOCH),desc='train'):
        model.train()
        _loss = []
        for X_local, X_global, X_frt, Y in train_dl:
            output, _ = model(X_local, X_global, X_frt)
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
            for x_local, x_global, x_frt, y in val_dl:
                model.eval()
                with torch.no_grad():
                    pred, _ = model(x_local, x_global, x_frt)
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
    test_ds = NestedDataset([
        torch.from_numpy(X_test[0]).float().to(device),
        torch.from_numpy(X_test[1]).float().to(device),
        torch.from_numpy(X_test[2]).float().to(device),
        torch.from_numpy(Y_test).float().to(device)
    ])
    test_dl = DataLoader(test_ds, batch_size=1)
    peak_idx = peak_visualization(Y_test=Y_test,points_near=points_near,pin=pin)
    model.to(device)
    model.eval()
    Y_pred = np.array([])
    with torch.no_grad():
        for x_local, x_global, x_frt, _ in test_dl:
            _y_pred, _ = model(x_local, x_global, x_frt)
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