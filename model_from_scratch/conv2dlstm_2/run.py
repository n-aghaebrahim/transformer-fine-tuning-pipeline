# Import necessary libraries and modules
import os
import json
import math
import plot

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Additional imports
from numpy import array
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta, datetime
from keras.preprocessing.sequence import TimeseriesGenerator
from contextlib import redirect_stdout

# Custom module imports
from indicators import INDICATOR
import decision
import config_run
import train_predict
import model_desing
import data_gen


# Main function
def main(
    stock_name: str=None, 
    epochs_num: int=None,
    n_steps: list=None,
    n_features: int=None,  
    train_condition: bool=None,  
    predict_condition: bool=None, 
    predict_date: str=None, 
    predict_time: str=None, 
    update_condition: bool=None, 
    prediction_day_adj_condition: bool=None, 
    prediction_day_adj: int=None, 
    today_date: str=None, 
    date_info: str=None, 
    now: str=None,   
    time: str=None, 
    about_model: str=None, 
    filter_num: int=None, 
    dilation: int=None,
    kernel: int=None,
    pad: str=None,
    pool: int=None, 
    lstm_layer_neuron_number: list=None,
    lstm_n_last_layer: int=None,
    last_layer_dropout: float=None,
    learning_rate_update: float=None,
    learning_rate: float=None,
    validation_split: float=None, 
    batch_size: int=None, 
    n_update_epoch: int=None, 
    best_weight_update: bool=None, 
    use_update_weight_use: bool=None,
    predict_weight: str=None,
    best_weight: bool=None,
    validation_split_update: float=None,
    lstm_layer_dict: dict=None,
):

    # Initialize a dictionary to store prediction results
    prediction_result = {}
    
    # Loop through each value of n_steps
    for s in n_steps:
        # Check if training is required and create a folder for logs
        if train_condition:
            if not os.path.exists(f"logs/{stock_name}/{date_info}/{time}/seq_{s}"):
                os.makedirs(f"logs/{stock_name}/{date_info}/{time}/seq_{s}")
        
            # Create an info.txt file and write general information to it
            with open(f"logs/{stock_name}/{date_info}/{time}/seq_{s}/info.txt", "w") as filet:
                filet.write("----------------------------\n")
                filet.write("general information\n")
                filet.write("----------------------------\n")
                filet.write(f"stock name: {stock_name}\n")
                filet.write(f"number of train epochs: {epochs_num}\n")
                filet.write(f"number of features: {n_features}\n")
                filet.write(f"the sequences that trained on at this time: {n_steps}\n")

        # Print a message for the current sequence
        print(
            f"\n\n\n\n\n********************* run for the sequence of {s} *************************"
        )

        # Call the function to get the data for the current sequence
        stock_data, values_all_a = data_gen.get_data(stock_name, date_info, time, train_condition)

        # Prepare the train and test datasets
        (
            X_train,
            y_train,
            X_test,
            y_test,
            scaled_predict,
            y_predict,
            pscaler,
            ypscaler,
            df_target,
        ) = data_gen.prepare_train_test(
            stock_data,
            values_all_a,
            train_condition,
            predict_condition,
            update_condition,
            s,
            n_features,
            prediction_day_adj_condition,
            prediction_day_adj,
            stock_name,
            date_info,
            time,
            validation_split,
            validation_split_update
        )
        
        # Initialize the model based on the provided configurations
        model = model_desing.MODEL(
            s,
            n_features,
            about_model,
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
        )

        # Train the model and obtain the updated model
        model = train_predict.model_train(
            stock_name,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs_num,
            train_condition,
            predict_condition,
            update_condition,
            s,
            stock_name,
            date_info,
            time,
            predict_time,
            predict_date,
            learning_rate_update,
            learning_rate,
            validation_split,
            batch_size,
            predict_weight,
            best_weight,
            best_weight_update,
            use_update_weight_use,
            n_update_epoch,
            validation_split_update
        )

        # Predict on the trained model and store the results in prediction_result
        print("\n############################################")
        print(f"prediction result for sequence of {s}......")
        print("############################################")
        result = train_predict.prediction(
            model, scaled_predict, y_predict, stock_data, ypscaler
        )
        prediction_result[s] = result

    # Print the target value (df_target) and make a decision based on prediction results
    print("\n\n##########################################")
    print("         the target value        ")
    print("##########################################\n")
    print(df_target)
    decision.make_decision(prediction_result)

    # Print the prediction results for each sequence
    for k, v in prediction_result.items():
        print(f"\n\nthe result for the sequence {k} is:\n")
        print(v.tail())


# Call the main function with the provided configurations from config_run module
main(
    stock_name = config_run.STOCK_NAME, 
    epochs_num = config_run.EPOCH_NUM,
    n_steps = config_run.N_STEPS,
    n_features = config_run.N_FEATURES,  
    train_condition = config_run.TRAIN_CONDITION,  
    predict_condition = config_run.PREDICT_CONDITION, 
    predict_date = config_run.PREDICT_DATE, 
    predict_time = config_run.PREDICT_TIME, 
    update_condition = config_run.UPDATE_CONDITION, 
    prediction_day_adj_condition = config_run.PREDICTION_DAY_ADJ_CONDITION, 
    prediction_day_adj = config_run.PREDICTION_DAY_ADJ, 
    today_date = config_run.TODAY_DATE, 
    date_info = config_run.DATE_INFO, 
    now = config_run.NOW,   
    time = config_run.TIME, 
    about_model = config_run.ABOUT_MODEL, 
    filter_num = config_run.FILTER_NUM, 
    dilation = config_run.DILATION,
    kernel = config_run.KERNEL,
    pad = config_run.PAD,
    pool = config_run.POOL, 
    lstm_layer_neuron_number = config_run.LSTM_LAYER_NEURON_NUMBER,
    lstm_n_last_layer = config_run.LSTM_N_LAST_LAYER,
    last_layer_dropout = config_run.LAST_LAYER_DROPOUT,
    learning_rate_update = config_run.LEARNING_RATE_UPDATE,
    learning_rate = config_run.LEARNING_RATE,
    validation_split = config_run.VALIDATION_SPLIT, 
    batch_size = config_run.BATCH_SIZE, 
    n_update_epoch = config_run.N_UPDATE_EPOCH, 
    best_weight_update = config_run.BEST_WEIGHT_UPDATE, 
    use_update_weight_use = config_run.USE_UPDATE_WEIGHT_USE,
    predict_weight = config_run.PREDICT_WEIGHT,
    best_weight = config_run.BEST_WEIGHT,
    validation_split_update = config_run.VALIDATION_SPLIT_UPDATE,
    lstm_layer_dict = config_run.LSTM_LAYER_DICT,
)
