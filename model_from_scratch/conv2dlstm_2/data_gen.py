import os
import json
import math
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from numpy import array
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool
from datetime import date, timedelta, datetime
from keras.preprocessing.sequence import TimeseriesGenerator
from contextlib import redirect_stdout
pd.options.mode.chained_assignment = None

import plot
from config_run import DATA_SELECTION, START_DATE_FOR_CAPTURING_DATA
from indicators import INDICATOR




def get_data(symbol, date_info, time, train_condition):
    stock_name = symbol
    #symbol = 'SPXU'
    today_date = date.today()
    next_date = today_date + timedelta(days=1)
    start_date = START_DATE_FOR_CAPTURING_DATA
    stock_data = yf.download(symbol, start=start_date, end=next_date, interval="1d")
    values_all = stock_data.values
    values_all_a = np.array(values_all)
    print('###### len data is: ', len(stock_data))
    #Calculate the On Balance Volume (OBV)
    if DATA_SELECTION['on_balance_volume']:
        stock_data =  INDICATOR().obv(stock_data=stock_data, span_period=[10,20, 30] )
        print('After OBV:', stock_data)


    # Calculate the ADX
    if DATA_SELECTION['adx']:
        stock_data = INDICATOR().get_adx(stock_data = stock_data)
        print('After ADX:', stock_data)

    # The Accumulation/Distribution indicatior
    if DATA_SELECTION['accumulation_distribution_indicator']:
        stock_data = INDICATOR().a_d(stock_data) 
        print('After AD:', stock_data)

    # Generate the ewm coloumns 
    if DATA_SELECTION['ewm']['condition']:
        stock_data = INDICATOR().ewm(stock_data, DATA_SELECTION['ewm']['lenght'])
        print('After EWM:', stock_data)

    # generate the macd coloumns
    if DATA_SELECTION['macd']:
        stock_data = INDICATOR().macd(stock_data) 
        print('After MACD:', stock_data)

    # generate stochastic oscillator indicator
    if DATA_SELECTION['stochastic_oscillator_indicator']:
        stock_data = INDICATOR().soi(stock_data)
        print('After SOI:', stock_data)

    # generate the rsi
    if DATA_SELECTION['rsi']:
        stock_data = INDICATOR().rsi(df=stock_data)
        print('After RSI:', stock_data)

    # calculate the bollinger bands
    if DATA_SELECTION['bollinger_bands']:
        stock_data = INDICATOR().get_bollinger_bands(stock_price=stock_data)
        print('After Bollinger Bands:', stock_data)

    print('44### test is: ', stock_data)
    # Converting the data frame values to the list
    stock_data_list = stock_data.values.tolist()
    print('### test is: ', stock_data)
    stock_data = stock_data.drop('Adj Close', axis=1)
    print('### test is: ', stock_data)
    today = str(stock_data.index[-1]).split(' ')[0]
    print('###### len data is: ', len(stock_data))
    # Check to see if it's a training/prediction run
    if train_condition: #not predict_condition:
        # Create the csv_data folder to save the csv data
        if not os.path.exists(f'logs/{stock_name}/{date_info}/{time}/csv_data'):
            os.makedirs(f'logs/{stock_name}/{date_info}/{time}/csv_data')

        # check and create a directory to save the stock data information
        if os.path.isfile(f'logs/{stock_name}/{date_info}/{time}/csv_data/data.csv'):
            data = pd.read_csv(f'logs/{stock_name}/{date_info}/{time}/csv_data/data.csv')

            print('-----------------------------------------------------')
            print(f'the last 10 days stock information for the {symbol}')
            print('-----------------------------------------------------\n')
            print(data.tail(10))
        	
            try:
                last_day = data['Date'].iloc[-1]
            except:
                last_day = data['Datetime'].iloc[-1]

            print('\n\n---------------------')
            print('general information')
            print('---------------------\n')
            print('today date: ', today)
            print('\nyesterday date: ', last_day)
        
            if last_day == today:
                print('\ncsv data status: no new information, same day')

            else:
                print('\ndata status: new information added, new day')

                stock_data.to_csv(f'logs/{stock_name}/{date_info}/{time}/csv_data/data.csv')


        else:
            stock_data.to_csv(f'logs/{stock_name}/{date_info}/{time}/csv_data/data.csv')
            data = pd.read_csv(f'logs/{stock_name}/{date_info}/{time}/csv_data/data.csv')
            print('-----------------------------------------------------')
            print(f'the last 10 days stock information for the {symbol}')
            print('-----------------------------------------------------\n')
            print(data.tail(10))
 
    
        # plot the lag adn autocorrelation for analysing the input data
        plot.lag_fig(stock_data, stock_name, date_info, time)
        plot.autocorrelation(stock_data, stock_name, date_info, time)

    else:
        data = stock_data
        print('-----------------------------------------------------')
        print(f'the last 10 days stock information for the {symbol}')
        print('-----------------------------------------------------\n')
        print(data.tail(10))


    # Converting the data frame to the numpy array
    stock_data = np.array(stock_data_list).astype('float32')

    return stock_data, values_all_a



# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps

		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break

		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)



# prepare the test and train data set
def prepare_train_test(stock_data, 
                       values_all_a, 
                       train_condition, 
                       predict_condition, 
                       update_condition, 
                       n_steps, 
                       n_features, 
                       prediction_day_adj_condition, 
                       prediction_day_adj, 
                       stock,
                       date_info,
                       time,
                       validation_split,
                       validation_split_update):


    #test_size = int(len(stock_data) * 0.008) # the test data will be 10% (0.1) of the entire data
    #print('test size is: ', test_size)

    #if not predict_condition:
    #    with open(f'logs/{stock}/{date_info}/{time}/seq_{n_steps}/info.txt', 'a') as filet:
    #        filet.write(f'test size: {test_size}\n')


    # dividing the data to do test and training in respect to the start day and end day for each
    # chosing the size of test data in respect the n_steps that it's training on
    if n_steps<=10:
        start_day_test = 5 * n_steps

    elif 10 < n_steps <=100 :
        start_day_test = 3 * n_steps

    else:
        start_day_test = 2*n_steps

    #end_day_test = start_day_test - 10
    #start_day_train_update = n_steps
    #end_day_train = 40
    #predict = True

    # df_target defined as None if the model doesn't do prediciton
    df_target = None


    if predict_condition:
        print('\nthe model is going to make prediction...')
        train = stock_data[:-start_day_test]
        train_update = stock_data[-n_steps:]
        test = stock_data[-start_day_test:]

        if prediction_day_adj_condition:
            predict_input = stock_data[-n_steps-prediction_day_adj:-prediction_day_adj] ##

        else:
            predict_input = stock_data[-n_steps:] ##


        print('\n\n--------------------------------------------------------------------')
        print('the last day information on test data set that uses for prediction:')
        print('-------------------------------------------------------------------')
        predict_reshape = predict_input[-1].reshape(1,predict_input[-1].shape[0])

        df_predict_reshape = pd.DataFrame(predict_input)#, columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'ewm5', 'ewm10', 'ewm20', 'ewm50', 'ewm200', 'macd', 'k', 'd'])  
        print(df_predict_reshape)
    
        print('\n\n------------------------------')
        print('the predict target values are:')
        print('------------------------------')
        if prediction_day_adj_condition:
            predict_target_reshape = stock_data[-prediction_day_adj].reshape(1, stock_data[0].shape[0])#,stock_data[-end_day_test].shape[0])
            df_predict_target_reshape = pd.DataFrame(predict_target_reshape)#, columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'ewm5', 'ewm10', 'ewm20', 'ewm50', 'ewm200', 'macd', 'k', 'd'])
            df_target = df_predict_target_reshape.iloc[:, :4]#.columns = ['Open', 'High', 'Low', 'Close']
            df_target.columns = ['Open', 'High', 'Low', 'Close']
            print(df_target)

        else:
            print('the target values do not exist since it wil be for a next day in future')
    


    if update_condition:
        print('\nthe model is going to train on recent data...')
        train = stock_data[:-start_day_test]
        train_update = stock_data[-start_day_test:] ## on recent data
        test = stock_data[-start_day_test:]
        predict_input = stock_data[-n_steps:]
        print('len train data for updating the model is: ', len(train_update))
        print('\n--------------------')
        print('train_update shape')
        print('--------------------')
        print(train_update.shape)
        print('len train_update data set is: ', len(train_update))
        print('len validation data is: ', len(train_update)*validation_split_update)


    if train_condition:
        
        print('\nthe model is going to train on the whole data set...')
        train = stock_data[:-start_day_test]  ##
        train_update = stock_data[-n_steps:]
        test = stock_data[-start_day_test:]   ##
        predict_input = stock_data[-n_steps:]
        print('len total data set is: ', len(stock_data))
        print('len train data set is: ', len(train))
        #print('len validation data is: ', len(train)*validation_split)
        print('len test data is: ', len(test))

        print('\n--------------------')
        print('train, test shape')
        print('--------------------')
        print(train.shape, test.shape)
    

    ################################################
    # Scale the train, validation and test dataset #
    ################################################

    ########################
    # scale the train data #
    ########################
    tscaler = MinMaxScaler(feature_range=(0,1))
    tscaler.fit(train)
    scaled_t_train = tscaler.transform(train)
    X_train, y_train = split_sequences(scaled_t_train, n_steps)
    X_train = X_train.reshape((X_train.shape[0], n_steps, 1, n_features, 1))
    #X_train = X_train.reshape((X_train.shape[0], 1, n_steps, n_features))
    new_y_train = []
    for i in y_train:
        new_y_train.append(i[:4])
    new_y_train = np.array(new_y_train)
    y_train = new_y_train

    if train_condition:
        print('\n-----------------------------------------------')
        print('X_train shape after converting to the seqences:')
        print('-----------------------------------------------')
        print('x train shape: ', X_train.shape)
        print('y train shape: ', y_train.shape)


    #######################
    # scale the test data #
    #######################
    ttscaler = MinMaxScaler(feature_range=(0,1))
    ttscaler.fit(test)
    scaled_test = ttscaler.transform(test)
    X_test, y_test = split_sequences(scaled_test, n_steps)
    new_y_test = []
    for ii in y_test:
        new_y_test.append(ii[:4])
    new_y_test = np.array(new_y_test)
    y_test = new_y_test
    X_test = X_test.reshape((X_test.shape[0], n_steps, 1, n_features,1))
    #X_test = X_test.reshape((X_test.shape[0],1, n_steps, n_features))
    if train_condition:
        print('\n-----------------------------------------------')
        print('test shape after converting to the seqences:')
        print('-----------------------------------------------')
        #print('x test shape: ', X_test.shape)
        print('y test shape: ', y_test.shape)  
        print('x test shape after reshape: ', X_test.shape)

    ###############################
    # scale the train update data #
    ##############################
    tscaler = MinMaxScaler(feature_range=(0,1))
    tscaler.fit(train_update)
    scaled_t_train_update = tscaler.transform(train_update)
    X_train_update, y_train_update = split_sequences(scaled_t_train_update, n_steps)
    X_train_update = X_train_update.reshape((X_train_update.shape[0], n_steps, 1, n_features, 1))
    #X_train_update = X_train_update.reshape((X_train_update.shape[0], 1, n_steps, n_features))

    if update_condition:
        print('\n-----------------------------------------------')
        print('X_train_update shape after converting to the seqences:')
        print('-----------------------------------------------')
        print('x train shape: ', X_train_update.shape)
        print('y train shape: ', y_train_update.shape)

    #############################
    # scale the prediction data #
    #############################
    pscaler = MinMaxScaler(feature_range=(0,1))
    pscaler.fit(predict_input)
    scaled_predict = pscaler.transform(predict_input)
    scaled_predict = scaled_predict.reshape(1, scaled_predict.shape[0], scaled_predict.shape[1])
    
    ypscaler = MinMaxScaler(feature_range=(0,1))

    new_y_pred = []
    for ii in predict_input:
        new_y_pred.append(ii[:4])
    new_y_pred = np.array(new_y_pred)
    print(new_y_pred[:4].shape)
    new_y_pred = np.array(new_y_pred)
    ypscaler.fit(new_y_pred)
    #X_predict, y_predict = split_sequences(scaled_predict, n_steps)
    y_predict = None
    scaled_predict = scaled_predict.reshape((scaled_predict.shape[0],  n_steps, 1, n_features, 1))
    #scaled_predict = scaled_predict.reshape((scaled_predict.shape[0],1, n_steps, n_features))
    if predict_condition:
        print('\n-----------------------------------------------')
        print('predict data shape after converting to the seqences:')
        print('-----------------------------------------------')
        print('x predict shape after reshape: ', scaled_predict.shape)
      

    return X_train, y_train, X_test, y_test, scaled_predict, y_predict, pscaler, ypscaler, df_target
