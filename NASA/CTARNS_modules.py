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

class DilatedCNN(nn.Module):
    def __init__(self, H, W, pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation):
        super().__init__()

        self.pad = pcnn_padding
        self.conv1d = nn.Conv1d(H, W, pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation)

    def forward(self, X):
        X = self.conv1d(X)
        return X[:,:,:-self.pad]

def masked_softmax(X, valid_lens, using_gumble=False, tau=1, hard=False, eps=1e-10, dim=- 1):
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
    def __init__(self, dropout, num_heads=None, using_gumble=False):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads  # To be covered later
        self.using_gumble = using_gumble

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
        self.attention_weights = masked_softmax(scores, valid_lens, using_gumble=self.using_gumble)
        return torch.bmm(self.dropout(self.attention_weights), values), self.attention_weights
        # return torch.bmm(self.dropout(self.attention_weights), values)

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
        
        self.P = torch.zeros(())
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

class TransformerEncoderBlock(nn.Module):
    @staticmethod
    def calc_cnn_output_shape(
        dim, ksize, stride=1, padding=0, dilation=1
    ):
        def shape_each_dim(i):
            odim_i = dim[i] + 2 * padding - dilation * (ksize - 1) - 1
            return int(odim_i / stride) + 1
        return shape_each_dim(0), shape_each_dim(1)

    def __init__(self, idim, dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation, dropout, 
                pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation):
        super().__init__()
        H, W = idim[0], idim[1]
        
        self.Wq = torch.nn.Parameter(torch.randn(H, H))
        self.Wk = torch.nn.Parameter(torch.randn(H, H))
        self.Wv = torch.nn.Parameter(torch.randn(H, H))
        self.attention = DotProductAttention(dropout)

        self.attention_weights = None

        self.attention_ln = AddNorm((H, W), dropout)
        
        self.pcnn1 = nn.Conv1d(H, H, pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation)
        lnshape = TransformerEncoderBlock.calc_cnn_output_shape((H, W), pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation)
        self.pcnn_ln1 = AddNorm([*lnshape], dropout)
        
        self.pcnn2 = nn.Conv1d(H, H, pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation)
        lnshape = TransformerEncoderBlock.calc_cnn_output_shape(lnshape, pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation)
        self.pcnn_ln2 = AddNorm([*lnshape], dropout)
        
        lnshape = TransformerEncoderBlock.calc_cnn_output_shape(lnshape, dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation)
        self.dccnn1 = nn.Conv1d(lnshape[0], lnshape[0], dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation)
        
        lnshape = TransformerEncoderBlock.calc_cnn_output_shape(lnshape, dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation)
        self.dccnn2 = nn.Conv1d(lnshape[0], lnshape[0], dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation)
    
    def forward(self, X):
        #input (B, H, W) 
        Q = torch.einsum('jj,ijk->ijk', self.Wq, X)
        K = torch.einsum('jj,ijk->ijk', self.Wk, X)
        V = torch.einsum('jj,ijk->ijk', self.Wv, X)
        Y, self.attention_weights = self.attention(Q,K,V)
        # Y = self.attention(Q,K,V)
        X = self.attention_ln(X, Y)
        X = self.pcnn_ln1(X, self.pcnn1(X))
        X = self.pcnn_ln2(X, self.pcnn2(X))
        X = self.dccnn1(X)
        X = self.dccnn2(X)
        return X, self.attention_weights
        # return X

class TransformerDecoderBlock(nn.Module):
    @staticmethod
    def calc_cnn_output_shape(
        dim, ksize, stride=1, padding=0, dilation=1
    ):
        def shape_each_dim(i):
            odim_i = dim[i] + 2 * padding - dilation * (ksize - 1) - 1
            return int(odim_i / stride) + 1
        return shape_each_dim(0), shape_each_dim(1)

    def __init__(self, idim, dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation, dropout, 
                pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation, using_gumble=False):
        super().__init__()
        H, W = idim[0], idim[1]

        self.dccnn1 = DilatedCNN(H, H, dccnn_kernel_size, dccnn_stride, 2*dccnn_padding, dccnn_dilation)
        lnshape = TransformerDecoderBlock.calc_cnn_output_shape((H,W), dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation)
        self.dccnn_ln1 = AddNorm([*lnshape], dropout)
        
        self.dccnn2 = DilatedCNN(lnshape[0], lnshape[0], dccnn_kernel_size, dccnn_stride, 2*dccnn_padding, dccnn_dilation)
        lnshape = TransformerDecoderBlock.calc_cnn_output_shape(lnshape, dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation)
        self.dccnn_ln2 = AddNorm([*lnshape], dropout)

        self.Wq = torch.nn.Parameter(torch.randn(lnshape[0], lnshape[0]))
        self.Wk = torch.nn.Parameter(torch.randn(lnshape[0], lnshape[0]))
        self.Wv = torch.nn.Parameter(torch.randn(lnshape[0], lnshape[0]))
        # self.Wq = nn.Linear(lnshape[1],lnshape[1])
        # self.Wk = nn.Linear(lnshape[1],lnshape[1])
        # self.Wv = nn.Linear(lnshape[1],lnshape[1])

        self.attention = DotProductAttention(dropout,using_gumble=using_gumble)
        self.attention_weight = None
        self.attention_ln = AddNorm((lnshape[0], lnshape[1]), dropout)
        
        self.pcnn1 = nn.Conv1d(lnshape[0], lnshape[0], pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation)
        lnshape = TransformerDecoderBlock.calc_cnn_output_shape(lnshape, pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation)
        # self.pcnn_ln1 = AddNorm([*lnshape], dropout)
        
        self.pcnn2 = nn.Conv1d(lnshape[0], lnshape[0], pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation)
        lnshape = TransformerDecoderBlock.calc_cnn_output_shape(lnshape, pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation)
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
        # return X

class Encoder(nn.Module):
    def __init__(self, idim, dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation, dropout, 
                pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation, n_layers):
        super().__init__()
        
        self.layers = nn.ModuleList([TransformerEncoderBlock(
            idim, 
            dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation, 
            dropout, 
            pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation
        ) for _ in range(n_layers)])
        self.attention_weights = []
    def forward(self, X):
        self.attention_weights = []
        for layer in self.layers:
            X, attention_weight = layer(X)
            # X = layer(X)
            self.attention_weights.append(attention_weight)
        
        return X, self.attention_weights
        # return X

# class Decoder(nn.Module):
#     def __init__(self, num_var, in_dim, out_dim):
#         super().__init__()
#         self.proj = nn.Linear(in_dim, out_dim)
#         self.linear = nn.Linear(num_var*out_dim,1)
#     def forward(self, X):
#         X = self.proj(X)
#         X = X.view(X.size(0),-1)
#         return self.linear(X)

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

class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, idim, dropout, encoder, decoder, post_decoder):
        super().__init__()
        self.pos_emb = PositionalEncoding(idim[1],dropout)
        self.encoder = encoder
        self.decoder = decoder
        self.post_decoder = post_decoder

    def forward(self, X):
        X = self.pos_emb(X)
        enc_outputs = self.encoder(X)
        dec_outputs = self.decoder(X, enc_outputs)
        # Return decoder output only
        return self.post_decoder(dec_outputs)

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

class CNN_Transformer_ARNS(nn.Module):
    def __init__(self, 
                 local_feature_size, num_extracted_point, window_size, global_feature_size, frt_feature_size, output_node,
                 dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation, 
                 dropout, 
                 pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation, 
                 local_n_layers, global_n_layers, local_using_gb=False, global_using_gb=False):
        super().__init__()
        self.local_decoder_list = nn.ModuleList()
        self.local_post_decoder_list = nn.ModuleList()
        for _ in range(window_size):
            decoder = Decoder(
                (local_feature_size, num_extracted_point), 
                dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation, 
                dropout, 
                pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation,
                local_n_layers, using_gumble=local_using_gb)

            post_decoder = Post_Decoder(None, num_extracted_point, 1)
#             model = nn.Sequential(decoder, post_decoder)
#             self.local_decoder_list.append(model)
            self.local_decoder_list.append(decoder)
            self.local_post_decoder_list.append(post_decoder)
            
        self.global_decoder = decoder = Decoder(
                                        (local_feature_size + global_feature_size, window_size), 
                                        dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation, 
                                        dropout, 
                                        pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation,
                                        global_n_layers, using_gumble=global_using_gb)
        
        self.attention_weights = {}

        self.post_decoder = Post_Decoder(None, window_size, 1)
        
        self.global_linear_layer = nn.Linear(global_feature_size, output_node)
        
        self.dense1 = nn.Linear(local_feature_size + global_feature_size + window_size*output_node + frt_feature_size, 10)
        self.tanh = nn.Tanh()
        self.dense2 = nn.Linear(10, 1)
        
    def forward(self, X_local, X_global, X_frt):
        self.attention_weights = {}
        self.attention_weights['local'] = []
        self.attention_weights['global'] = []
        X_local = torch.reshape(X_local, (X_local.size(0),X_local.size(1),X_local.size(3),X_local.size(2)))
        Y = []
        for i in range(len(self.local_decoder_list)):
            y, attention_weights = self.local_decoder_list[i](X_local[:,i,:,:])
            # y = self.local_decoder_list[i](X_local[:,i,:,:])
            y = self.local_post_decoder_list[i](y)
            Y.append(y)
            self.attention_weights['local'].extend(attention_weights)
        Y = torch.cat(Y,axis=-1)
        
        global_linear = self.global_linear_layer(X_global)
        
        global_linear = global_linear.view(global_linear.size(0), -1)
        
        Y = torch.cat((Y, torch.reshape(X_global, (-1,X_global.size(-1),X_global.size(-2)))),axis=1)
        
        Y, attention_weights = self.global_decoder(Y)
        # Y = self.global_decoder(Y)
        
        self.attention_weights['global'].extend(attention_weights)
        
        Y = self.post_decoder(Y)
        
        Y = Y.view(Y.size(0), -1)
        
        Y = torch.cat([Y, global_linear, X_frt],axis=-1)
        
        return self.dense2(self.tanh(self.dense1(Y))), self.attention_weights
        # return self.dense2(self.tanh(self.dense1(Y)))
        
        