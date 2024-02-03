from ARNS import ARNS
from CTARNS_modules import CNN_Transformer_ARNS
import numpy as np
import functools

import torch.nn.functional as F
import math

import torch.nn.functional as F
import torch.nn as nn
import torch
from utils import setup_seed, evaluation, relative_error, denormalize, peak_visualization, mean_absolute_percentage_error,root_mean_square_error
import os
import matplotlib.pyplot as plt
import time
import copy

class CTARNS(ARNS):
    def build_model(self, local_feature_size=3, num_extracted_point=11, window_size=8, global_feature_size=3, frt_feature_size=1, output_node=1,dccnn_kernel_size=3, dccnn_stride=1, dccnn_padding=1, dccnn_dilation=1, dropout=0.01, pcnn_kernel_size=1, pcnn_stride=1, pcnn_padding=0, pcnn_dilation=1, n_layers=2):
        return CNN_Transformer_ARNS(local_feature_size, num_extracted_point, window_size, global_feature_size, frt_feature_size, output_node,dccnn_kernel_size, dccnn_stride, dccnn_padding, dccnn_dilation, dropout, pcnn_kernel_size, pcnn_stride, pcnn_padding, pcnn_dilation, n_layers)

    def train(self, model, X_train, Y_train, X_val, Y_val, out_dir, EPOCH=1000, LR=0.001, seed=0, weight_decay=0., device="cpu", clipping_value=None, verbose=2, patience=25):
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        print("Start training")
        
        loss_list, y_ = [0], [] 
        rmse_list, mape_list = [], []
        best_rmse = None

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        R_avg = []
        for epoch in range(EPOCH):
            model.train()
            X_local, X_global, X_frt = X_train
            X_local = np.array(X_local).astype(np.float32)
            X_global = np.array(X_global).astype(np.float32)
            X_frt = np.array(X_frt).astype(np.float32)
            Y = np.array(Y_train).astype(np.float32)
            X_local, X_global, X_frt = torch.from_numpy(X_local).to(device), torch.from_numpy(X_global).to(device), torch.from_numpy(X_frt).to(device)
            Y = torch.from_numpy(Y).to(device)
            # output, attention_weights = model(X_local, X_global, X_frt)
            output = model(X_local, X_global, X_frt)
            downstream_loss = criterion(output, Y)
            loss = downstream_loss

            optimizer.zero_grad()
            loss.backward()
            if clipping_value:
                nn.utils.clip_grad_norm(model.parameters(), clipping_value)
            optimizer.step()

            if epoch%verbose==0 or epoch==(EPOCH-1):
                point_list = []
                x_local, x_global, x_frt = X_val
                x_local = np.array(x_local).astype(np.float32)
                x_global = np.array(x_global).astype(np.float32)
                x_frt = np.array(x_frt).astype(np.float32)
                x_local, x_global, x_frt = torch.from_numpy(x_local).to(device), torch.from_numpy(x_global).to(device), torch.from_numpy(x_frt).to(device)
                model.eval()
                with torch.no_grad():
                    # pred, _ = model(x_local, x_global, x_frt)
                    pred = model(x_local, x_global, x_frt)
                point_list = list(pred[:,0].cpu().detach().numpy())
                y_.append(point_list)
                loss_list.append(loss)
                mape, rmse = evaluation(y_test=denormalize(Y_val).flatten(), y_predict=denormalize(np.array(y_[-1])))
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
                
            if (len(loss_list) > 1) and (abs(loss_list[-2] - loss_list[-1]) < 1e-7):
                break

        torch.save(best_rmse['model'], os.path.join(out_dir,'checkpoint.pt'))
        mape, rmse = evaluation(y_test=denormalize(Y_val).flatten(), y_predict=denormalize(np.array(y_[-1])))
        re = relative_error(y_test=Y_val.flatten(), y_predict=y_[-1], threshold=0.7)

        print("Train successfully")

        return model, y_[-1], mape_list, rmse_list, re

    def test(self, model, X_test, Y_test, out_dir, points_near=0, device='cpu', starting_point=0, pin=None):
        X_local, X_global, X_frt = X_test
        X_local = torch.from_numpy(X_local).float().to(device)
        X_global = torch.from_numpy(X_global).float().to(device)
        X_frt = torch.from_numpy(X_frt).float().to(device)
        Y_test = torch.from_numpy(Y_test).float().to(device)
        peak_idx = peak_visualization(Y_test=Y_test,points_near=points_near,pin=pin)
        model.to(device)
        model.eval()
        with torch.no_grad():
            # Y_pred, _ = model(X_local, X_global, X_frt)
            Y_pred = model(X_local, X_global, X_frt)
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
        return pred, test

