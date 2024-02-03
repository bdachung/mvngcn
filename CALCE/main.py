import numpy as np
from FXP import FXP
from ARNS import ARNS
from CTARNS import CTARNS
from basic_models import GF, Battery_LSTM, Battery_SVR, MLP
from utils import print_num_params
import time
import os
from MC_LSTM import MC_LSTM
from detranformer import DeTransformer
import torch

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
BATTERIES = ['CS2_35', 'CS2_36','CS2_37','CS2_38']
extract_points_len = 11

# arns = ARNS()
# os.mkdir('ARNS')

# for bat in BATTERIES:
#     batteries = BATTERIES.copy()
#     batteries.remove(bat)
#     batteries = batteries + [bat]

#     X_train, Y_train, X_val, Y_val, X_test,Y_test = arns.cross_validation(batteries, Battery, rest_time, validation=0.1)

#     model = arns.build_model((X_train[0].shape[1:], X_train[1].shape[1:], X_train[2].shape[1:]))

#     start = time.time()
#     arns.train(model, X_train, Y_train, X_val, Y_val, out_dir='ARNS/' + str(batteries[-1]),epochs=100)
#     end = time.time()

#     model.load_weights('ARNS/' + str(batteries[-1]) + '/checkpoint_weight')
#     arns.test(model, X_test, Y_test, out_dir='ARNS/' + str(batteries[-1]), pin=str(batteries[-1]))
#     with open('ARNS/' + str(batteries[-1]) + '/result.txt', 'a') as f:
#         f.write(f"Training time {end - start} seconds" +'\n')

# mc_lstm = MC_LSTM()
# os.mkdir('MC_LSTM')

# for bat in BATTERIES:
#     batteries = BATTERIES.copy()
#     batteries.remove(bat)
#     batteries = batteries + [bat]

#     X_train, Y_train, X_val, Y_val, X_test,Y_test = mc_lstm.cross_validation(batteries, Battery, rest_time, validation=0.1)

#     model = mc_lstm.build_model((X_train[0].shape[1:], X_train[1].shape[1:], X_train[2].shape[1:], X_train[3].shape[1:]))

#     start = time.time()
#     mc_lstm.train(model, X_train, Y_train, X_val, Y_val, out_dir='MC_LSTM/' + str(batteries[-1]),epochs=1000,verbose=1)
#     end = time.time()

#     model.load_weights('MC_LSTM/' + str(batteries[-1]) + '/checkpoint_weight')
#     mc_lstm.test(model, X_test, Y_test, out_dir='MC_LSTM/' + str(batteries[-1]), pin=str(batteries[-1]))
#     with open('MC_LSTM/' + str(batteries[-1]) + '/result.txt', 'a') as f:
#         f.write(f"Training time {end - start} seconds" +'\n')

# fxp = FXP()

# os.mkdir('FXP')

# for bat in BATTERIES:
#     batteries = BATTERIES.copy()
#     batteries.remove(bat)
#     batteries = batteries + [bat]

#     X_train, Y_train, X_val, Y_val, X_test,Y_test = fxp.cross_validation(batteries, Battery, rest_time, validation=0.1)

#     model = fxp.build_model(model_mode='bert-trainposcode', feature_mode='LSTM_bi', use_AR=True, num_labels=1, out_embed_dim=32, hidden_dim=32, dropout=0.05)
#     print_num_params(model)

#     start = time.time()
#     fxp.train(model, X_train, Y_train, X_val, Y_val, out_dir='FXP/' + str(batteries[-1]), EPOCH=3000)
#     end = time.time()

#     with open('FXP/' + str(batteries[-1]) + '/result.txt', 'a') as f:
#         f.write(f"Training time {end - start} seconds" +'\n')
    
#     model = torch.load('FXP/' + str(batteries[-1]) + '/checkpoint.pt')

#     fxp.test(model, X_test, Y_test, out_dir='FXP/' + str(batteries[-1]), pin=str(batteries[-1]))


# ctarns = CTARNS()
# os.mkdir('CTARNS')

# for bat in BATTERIES:
#     if bat in ['CS2_35']:
#         continue
#     batteries = BATTERIES.copy()
#     batteries.remove(bat)
#     batteries = batteries + [bat]

#     X_train, Y_train, X_val, Y_val, X_test,Y_test = ctarns.cross_validation(batteries, Battery, rest_time, validation=0.1)

#     model = ctarns.build_model()
#     print_num_params(model)

#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     start = time.time()
#     ctarns.train(model, X_train, Y_train, X_val, Y_val, out_dir='CTARNS/' + str(batteries[-1]),EPOCH=3000,LR=0.005,verbose=10)
#     end = time.time()

#     with open('CTARNS/' + str(batteries[-1]) + '/result.txt', 'a') as f:
#         f.write(f"Training time {end - start} seconds" +'\n')
#     # print(f"Training time {end - start} seconds")

#     model = torch.load('CTARNS/' + str(batteries[-1]) + '/checkpoint.pt')

#     ctarns.test(model, X_test, Y_test, out_dir='CTARNS/' + str(batteries[-1]), pin=str(batteries[-1]))

# gf = GF()
# os.mkdir('GFv1')

# for bat in BATTERIES:
#     batteries = BATTERIES.copy()
#     batteries.remove(bat)
#     batteries = batteries + [bat]

#     X_train, Y_train, X_val, Y_val, X_test,Y_test = gf.cross_validation(batteries, Battery, validation=0.1)

#     model = gf.build_model(k=5)
#     print_num_params(model)

#     start = time.time()
#     gf.train(model, X_train, Y_train, out_dir='GFv1/' + str(batteries[-1]),lr=0.05,epochs=50000)
#     end = time.time()
#     # print(f"Traing time {stop - start} seconds")
#     with open('GFv1/' + str(batteries[-1]) + '/result.txt', 'a') as f:
#         f.write(f"Training time {end - start} seconds" +'\n')

#     gf.test(model, X_test, Y_test, out_dir='GFv1/' + str(batteries[-1]), pin=str(batteries[-1]))

# svr = Battery_SVR()
# os.mkdir('SVR')

# for bat in BATTERIES:
#     batteries = BATTERIES.copy()
#     batteries.remove(bat)
#     batteries = batteries + [bat]

#     X_train, Y_train, X_val, Y_val, X_test,Y_test = svr.cross_validation(batteries, Battery, validation=0.1)

#     model = svr.build_model()

#     svr.train(model, X_train, Y_train, out_dir='SVR/' + str(batteries[-1]))
#     svr.test(model, X_test, Y_test, out_dir='SVR/' + str(batteries[-1]), pin=str(batteries[-1]))

# mlp = MLP()
# os.mkdir('MLP')

# for bat in BATTERIES:
#     batteries = BATTERIES.copy()
#     batteries.remove(bat)
#     batteries = batteries + [bat]

#     X_train, Y_train, X_val, Y_val, X_test,Y_test = mlp.cross_validation(batteries, Battery, rest_time, validation=0.1)

#     model = mlp.build_model(feature_size=8*11*3 + 8*3)

#     start = time.time()
#     mlp.train(model, X_train, Y_train, X_val, Y_val, out_dir='MLP/' + str(batteries[-1]), EPOCH=3000)
#     end = time.time()

#     with open('MLP/' + str(batteries[-1]) + '/result.txt', 'a') as f:
#         f.write(f"Training time {end - start} seconds" +'\n')

#     model  = torch.load('MLP/' + str(batteries[-1])+ '/checkpoint.pt')

#     mlp.test(model, X_test, Y_test, out_dir='MLP/' + str(batteries[-1]), pin=str(batteries[-1]))

# mlp = Battery_LSTM()

# os.mkdir('LSTM')

# for bat in BATTERIES:
#     batteries = BATTERIES.copy()
#     batteries.remove(bat)
#     batteries = batteries + [bat]

#     X_train, Y_train, X_val, Y_val, X_test,Y_test = mlp.cross_validation(batteries, Battery, rest_time, validation=0.1)

#     model = mlp.build_model(input_size=11*3 + 3, hidden_dim=32, num_layers=1, n_class=1, mode='LSTM')

#     start = time.time()
#     mlp.train(model, X_train, Y_train, X_val, Y_val, out_dir='LSTM/' + str(batteries[-1]), EPOCH=2000)
#     end = time.time()

#     with open('LSTM/' + str(batteries[-1]) + '/result.txt', 'a') as f:
#         f.write(f"Training time {end - start} seconds" +'\n')

#     model  = torch.load('LSTM/' + str(batteries[-1])+ '/checkpoint.pt')

#     mlp.test(model, X_test, Y_test, out_dir='LSTM/' + str(batteries[-1]), pin=str(batteries[-1]))

######### Detranformer
Rated_Capacity = 1.1
window_size = 8
feature_size = window_size
dropout = 0.0
EPOCH = 2000
nhead = 4
hidden_dim = 16
num_layers = 1
lr = 0.01    # learning rate
weight_decay = 0.0
noise_level = 0.0
alpha = 1e-5
is_load_weights = True
metric = 're'
seed = 0
global_seq_features = ['capacity']

detransformer = DeTransformer()

if not os.path.isdir('DeTransformer'):
    os.mkdir('DeTransformer')

for bat in BATTERIES:
    batteries = BATTERIES.copy()
    batteries.remove(bat)
    batteries = batteries + [bat]

    X_train, Y_train, X_val, Y_val, X_test,Y_test = detransformer.cross_validation(batteries, Battery, rest_time, extract_points_len=extract_points_len, multichannel_features=multichannel_features, window_size=window_size, prediction_interval=prediction_interval, global_seq_features = global_seq_features, key='cycle', label='capacity', random_state=7, shuffle=False, validation=0.1)

    model = detransformer.build_model(feature_size=feature_size, hidden_dim=hidden_dim, num_layers=num_layers, nhead=nhead, dropout=dropout, noise_level=noise_level)

    start = time.time()
    detransformer.train(model, X_train, Y_train, X_val, Y_val, out_dir='DeTransformer/' + str(batteries[-1]), LR=lr, EPOCH=3000, alpha=alpha, feature_size=feature_size)
    end = time.time()

    with open('DeTransformer/' + str(batteries[-1]) + '/result.txt', 'a') as f:
        f.write(f"Training time {end - start} seconds" +'\n')

    model  = torch.load('DeTransformer/' + str(batteries[-1])+ '/checkpoint.pt')

    detransformer.test(model, X_test, Y_test, out_dir='DeTransformer/' + str(batteries[-1]), pin=str(batteries[-1]), points_near=2)