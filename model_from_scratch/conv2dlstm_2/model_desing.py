# Import necessary libraries
import os
import json
import math
import plot
import pathlib

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from numpy import array
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta, datetime
from keras.preprocessing.sequence import TimeseriesGenerator
from contextlib import redirect_stdout
from tensorflow.keras.layers import Input, Reshape

# Import custom modules
from indicators import INDICATOR

pd.options.mode.chained_assignment = None

# Function to create a custom model
def MODEL(n_steps, 
          n_features, 
          about_model_summary, 
          filter_num, 
          dilation, 
          kernel, 
          pad, 
          pool, 
          lstm_layer_dict, 
          stock_name,
          date_info,
          time,
          lstm_n_last_layer,
          last_layer_dropout,
          train_condition
          ):

    # If train_condition is true, create and store model information
    if train_condition:
        if not pathlib.Path(f'logs/{stock_name}/{date_info}/{time}/seq_{n_steps}').is_dir():
            pathlib.Path('logs/{stock_name}/{date_info}/{time}/seq_{n_steps}').mkdir(parents=True)

        # Store model summary information in JSON format
        with open(f'logs/{stock_name}/{date_info}/{time}/seq_{n_steps}/info_dict.json', "w") as outfile:
            json.dump(about_model_summary, outfile, indent=6)

        # Store model configuration details in a text file
        with open(f'logs/{stock_name}/{date_info}/{time}/seq_{n_steps}/info.txt', 'a') as filet:
            filet.write('\n--------------------\n')
            filet.write('model information\n')
            filet.write('---------------------\n')
            filet.write(f'filter_num: {filter_num}\ndilation_rate: {dilation}\nkernel_size: {kernel}\npadding: {pad}\npool_size: {pool}\nlstm_num: {str(lstm_layer_dict)}\n')
        
    # Define the input layer for the model
    inputs = tf.keras.layers.Input(shape=(n_steps, n_features))
    inputs = Reshape((n_steps,1, n_features, 1))(inputs)

    # Apply Convolutional Neural Network (CNN) layers to the input data
    cnn = layers.TimeDistributed(tf.keras.layers.Conv2D(filters=filter_num, kernel_size=kernel, dilation_rate=dilation, padding=pad, activation='relu'))(inputs)
    cnn = layers.TimeDistributed(layers.Flatten())(cnn)

    # Create LSTM layers based on the configuration provided in lstm_layer_dict
    lstm_layer_list = []
    i = 0
    for k, v in lstm_layer_dict.items():
        if i == 0:
            lstm = layers.LSTM(v, activation='relu', return_sequences=True)(cnn)
            lstm = layers.Dropout(0.2)(lstm)
        else:
            lstm = layers.LSTM(v, activation='relu', return_sequences=True)(lstm)
            lstm = layers.Dropout(0.2)(lstm)
        i += 1

    lstm_out = layers.LSTM(lstm_n_last_layer, activation='relu')(lstm)
    lstm_out = layers.Dropout(last_layer_dropout)(lstm_out)

    # Define the output layer of the model
    out = layers.Dense(4)(lstm_out)  # Assuming 4 output values (Open, High, Low, Close), modify as needed

    # Create the final model
    model = keras.Model(inputs=inputs, outputs=out)

    # Print and store model summary if train_condition is true
    print('\n---------------------------')
    print('     * model summary *     ')
    print('---------------------------')
    if train_condition:
        with open(f'logs/{stock_name}/{date_info}/{time}/seq_{n_steps}/info_s.txt', 'w') as filet:
            with redirect_stdout(filet):
                model.summary()

    print('model is created')
    print(model.summary())
    return model
