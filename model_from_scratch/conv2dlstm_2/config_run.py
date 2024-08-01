import os
import pandas as pd
from datetime import date, timedelta, datetime



#########################
###### General Info #####
#########################
TODAY_DATE = date.today()
DATE_INFO = TODAY_DATE.strftime("%b-%d-%Y")
NOW = datetime.now()
TIME = NOW.strftime("%H:%M")



#########################
###### Run Condition ####
#########################
TRAIN_CONDITION = False
PREDICT_CONDITION = True
UPDATE_CONDITION = False





#########################
#### DATA Selection #####
#########################
DATA_SELECTION = {'on_balance_volume': True, 'adx': True, 'accumulation_distribution_indicator': True, 'ewm': {'condition': True, 'lenght': [2,3,5, 7, 9, 10, 15, 20, 30, 35, 40, 45, 50, 75, 130, 150, 200]},
                  'macd': True, 'stochastic_oscillator_indicator': True, 'rsi': True, 'bollinger_bands': True}

START_DATE_FOR_CAPTURING_DATA = '2021-01-01'
START_DATE_FOR_CAPTURING_DATA = '2023-07-28'

#########################
####### TRAIN ###########
#########################
STOCK_NAME = "TSLA"
EPOCH_NUM = 690#4000
N_STEPS = [2,3,4, 5,6,10,25, 30] #25,30,35,40,45,50,55,60,65,70]#, 9,10,11,12,13,14,15]#[2, 4, 5, 7, 8, 9, 10, 13, 14,15, 16, 18, 19]#[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] #[25, 50, 60, 80, 100, 120, 150, 200, 300, 400, 500] #[1, 2, 3, 4, 5, 10] #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20]




#########################
##### TRAIN/UPDATE ######
#########################

LEARNING_RATE = 0.0005
VALIDATION_SPLIT = 0.1
BATCH_SIZE = 16



#########################
####### UPDATE ##########
#########################
LEARNING_RATE_UPDATE = 0.0005
N_UPDATE_EPOCH = 100
BEST_WEIGHT_UPDATE = False

VALIDATION_SPLIT_UPDATE = 0.1








#########################
####### PREDICT #########
#########################
PREDICT_WEIGHT = '' #f'updated_{N_UPDATE_EPOCH}_'
BEST_WEIGHT = True #True
PREDICT_DATE = "Jul-30-2024"#"Feb-11-2023"#"Jan-04-2023"
PREDICT_TIME = "23:00"#"19.17"#"23:49"#"22:10"
USE_UPDATE_WEIGHT_USE = False
PREDICTION_DAY_ADJ_CONDITION = True
PREDICTION_DAY_ADJ = 1
DATA_SELECTION_SYMBOL = 'TSLA'



#########################
###### MODEL INFO #######
#########################
# check and create a directory to save the stock data information
if os.path.isfile(f'logs/{STOCK_NAME}/{PREDICT_DATE}/{PREDICT_TIME}/csv_data/data.csv') and PREDICT_CONDITION:
    data = pd.read_csv(f'logs/{STOCK_NAME}/{PREDICT_DATE}/{PREDICT_TIME}/csv_data/data.csv')
    data_features = data.columns
    N_FEATURES = len(data_features)

else:
    N_FEATURES = 39

FILTER_NUM = 16#16 #8#16
DILATION = 4 #None #3 
KERNEL = 3 #4
POOL = 2
PAD = "same" #"same"
LSTM_LAYER_NEURON_NUMBER = [500, 500, 400, 200] #[800, 1200, 1200, 600, 600, 200] #[600, 600, 600, 300, 300] #[800, 1200, 1200, 600, 600, 200]
N_LSTM_LAYER = len(LSTM_LAYER_NEURON_NUMBER)
LSTM_LAYER_DICT = {}

# Create a dictionary that contains all the n umber of neurons for each layer in lstm
for i in range(N_LSTM_LAYER):
    LSTM_LAYER_DICT[i] = LSTM_LAYER_NEURON_NUMBER[i]


LSTM_N_LAST_LAYER = 100
LAST_LAYER_DROPOUT = 0.05

ABOUT_MODEL_DICT = {
               'n_lstm_layer': N_LSTM_LAYER,
               'lstm_layer_neuron_number': LSTM_LAYER_NEURON_NUMBER,
               'lstm_n_last_layer': LSTM_N_LAST_LAYER,
               'last_layer_dropout': LAST_LAYER_DROPOUT,
               'pad': PAD,
               'pool': POOL,
               'kernel': KERNEL,
               'dilation': DILATION,
               'filter_num': FILTER_NUM,
               'n_features': N_FEATURES,
               'prediction_day_adj': PREDICTION_DAY_ADJ,
               'prediction_day_adj_condition': PREDICTION_DAY_ADJ_CONDITION,
               'predict_time': PREDICT_TIME,
               'predict_date': PREDICT_DATE,
               'validation_split_update': VALIDATION_SPLIT_UPDATE,
               'best_weight': BEST_WEIGHT,
               'predict_weight': PREDICT_WEIGHT,
               'use_update_weight_use': USE_UPDATE_WEIGHT_USE,
               'best_weight_update': BEST_WEIGHT_UPDATE,
               'n_update_epoch': N_UPDATE_EPOCH,
               'learning_rate_update': LEARNING_RATE_UPDATE,
               'learning_rate': LEARNING_RATE,
               'validation_split': VALIDATION_SPLIT,
               'batch_size': BATCH_SIZE,
               'stock_name': STOCK_NAME,
               'epoch_num': EPOCH_NUM,
               'n_steps': N_STEPS,
               'train_condition': TRAIN_CONDITION,
               'update_condition': UPDATE_CONDITION,
               'predict_condition': PREDICT_CONDITION,
               'data_selection': DATA_SELECTION,
               'start_date_for_capturing_data': '2015-01-01'
            }

ABOUT_MODEL = ABOUT_MODEL_DICT




########################################################
#### getting the information in case of existance ######
########################################################






