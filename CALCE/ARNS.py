import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, ConvLSTM2D, TimeDistributed, InputLayer, Input, Lambda, LSTM, Bidirectional
import matplotlib.pyplot as plt
import numpy as np
import math
from cs_interface import NestedModel
from utils import denormalize,mean_absolute_percentage_error,peak_visualization,root_mean_square_error
import os
import functools
from sklearn.model_selection import train_test_split

class ARNS(NestedModel):
    def build_model(self, input_shape, hidden_node_local_seq=5, hidden_node_global_seq=10, output_node=1, extract_points_len=11, multichannel_features=['voltage','current','resistance'], window_size=8, global_seq_features = ['x5', 'rest_period', 'capacity']):
        input_shape_mc, input_shape_glb, input_shape_frt = input_shape
    #     hidden_node_local_seq = 5
    #     hidden_node_global_seq = 10
    #     output_node = 1

        sample = input_shape_mc[0]
        mc_time_step = extract_points_len
        mc_feature = len(multichannel_features)
        glb_time_step = window_size
        glb_feature = len(global_seq_features)

        # Creating the layers
        input_layer_mc = Input(shape=input_shape_mc) # multichannel input (V,I,T)
        input_layer_glb = Input(shape=input_shape_glb) # global sequence input (x2, x3, ..., p, c, t_rest)
        input_layer_frt = Input(shape=input_shape_frt) # forecast cycle rest time input (t_rest)

        local_sequence_input = [tf.reshape(input_layer_mc[:,x,:,:], (-1, extract_points_len, len(multichannel_features))) for x in range(window_size)]

        # # local sequence return last node BiLSTM
        local_sequence_lstm_layers = [
        tf.reshape(
            tf.concat(
                axis=1,
                values=[
                        tf.reshape(Bidirectional(
                            LSTM(hidden_node_local_seq, input_shape=(sample, mc_time_step, mc_feature), return_sequences = False)
                        )(local_sequence_input[x]), (-1, 2 * hidden_node_local_seq)),
                        input_layer_glb[:,x,:]
                ]
            ),
            (-1, 1, 2*hidden_node_local_seq + glb_feature)
        ) for x in range(window_size)]

        global_sequence_input = tf.concat(axis=1,values=local_sequence_lstm_layers)
        
        print("global shape",global_sequence_input.shape)
        
        global_sequence_lstm_layer = Bidirectional(LSTM(hidden_node_global_seq, input_shape=(sample, glb_time_step, 2*hidden_node_local_seq*mc_time_step + glb_feature), return_sequences = False))(global_sequence_input)

        print("output global:",global_sequence_lstm_layer.shape)
        
        # # Final neural net
        linear_layer = Dense(output_node)(input_layer_glb)
        print("linear layer:", linear_layer.shape)
        output_layer = tf.concat(axis=1,values=[global_sequence_lstm_layer, input_layer_frt, linear_layer[:,:,0]])
        print("output layer:", output_layer.shape)
        output_layer = Dense(10, activation='tanh')(output_layer)
        output_layer = Dense(8, activation='tanh')(output_layer)
        output_layer = Dense(4, activation='tanh')(output_layer)
        output_layer = Dense(output_node)(output_layer)

        # Defining the model by specifying the input and output layers
        model = Model(inputs=[input_layer_mc, input_layer_glb, input_layer_frt], outputs=output_layer)
        model.compile(optimizer='adam',loss='mse',metrics=[tf.keras.metrics.RootMeanSquaredError()])
        print(model.summary())

        return model

    def extract_batter_features(self, df, global_df, battery_name, window_size=8, prediction_interval=1, extract_points_len=11, multichannel_features=['voltage','current','resistance'], global_seq_features = ['x5', 'rest_period', 'capacity'], key='cycle', label='capacity', rest_period='rest_period'):
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

    def cross_validation(self, batteries, df, rest_df, extract_points_len=11, multichannel_features=['voltage','current','resistance'], window_size=8, prediction_interval=1, global_seq_features = ['x5', 'rest_period', 'capacity'], key='cycle', label='capacity', rest_period='rest_period', validation=None, random_state=7, shuffle=False):
        # last pin is the test set
        all_batteries = [self.extract_batter_features(df, rest_df, battery_nb, window_size=window_size, prediction_interval=prediction_interval, extract_points_len=extract_points_len, multichannel_features=multichannel_features, global_seq_features=global_seq_features, key=key, label=label, rest_period=rest_period) for battery_nb in batteries]

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

    def train(self, model, X_train, Y_train, X_val, Y_val, out_dir, epochs=1000, batch_size=32, verbose=0):
        print("Start training")
        tf.keras.backend.clear_session()
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        # fit network
        checkpoint_filepath = out_dir + '/checkpoint_weight'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_root_mean_squared_error',
            mode='min',
            save_best_only=True)
        history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val), verbose=verbose, callbacks=[model_checkpoint_callback])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title("Model Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Valid'])
        
        plt.savefig(out_dir + '/history_loss.png', format='png')
        plt.close()
        # model.save(out_dir + '/checkpoint.h5', save_format="h5")
        print("Train successful. The checkpoint_weight is saved at " + out_dir)

        return model, history

    def test(self, model, X_test, Y_test, out_dir, starting_point=0, pin=None):
        print("Start predicting")
        Y_pred = model.predict(X_test)
        peak_idx = peak_visualization(Y_test=Y_test,points_near=0,pin=pin)
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
        plt.close()

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

        return pred, test