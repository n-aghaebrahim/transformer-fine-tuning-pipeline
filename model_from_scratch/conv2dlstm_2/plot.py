# Import necessary libraries
import os
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot, autocorrelation_plot

# Function to create a lag plot and save it as an image
def lag_fig(stock_data, stock_name, date_info, time):
    # Create a folder for data plots if it does not exist
    if not os.path.exists(f'logs/{stock_name}/{date_info}/{time}/data_plots'):
        os.makedirs(f'logs/{stock_name}/{date_info}/{time}/data_plots')

    # Generate a lag plot for the stock_data
    lag_plot(stock_data)

    # Save the lag plot as an image in the data_plots folder
    plt.savefig(f'logs/{stock_name}/{date_info}/{time}/data_plots/laag_plot.png')
    plt.clf()

# Function to create an autocorrelation plot and save it as an image
def autocorrelation(stock_data, stock_name, date_info, time):
    # Create a folder for data plots if it does not exist
    if not os.path.exists(f'logs/{stock_name}/{date_info}/{time}/data_plots'):
        os.makedirs(f'logs/{stock_name}/{date_info}/{time}/data_plots')

    # Generate an autocorrelation plot for the stock_data
    autocorrelation_plot(stock_data)

    # Save the autocorrelation plot as an image in the data_plots folder
    plt.savefig(f'logs/{stock_name}/{date_info}/{time}/data_plots/aautocorrelation_plot.png')
    plt.clf()

# Function to plot and save training/validation accuracy and loss
def plot_fig(history, s, stock_name, date_info, time, update_condition, predict_date, predict_time, n_update_epoch):
    # Create plot names based on the sequence and update condition
    plot_acc_name = f'acc_sequence_{s}'
    plot_loss_name = f'loss_sequence_{s}'
    if update_condition:
        time = predict_time
        date_info = predict_date
        plot_acc_name = plot_acc_name + f'_update_{n_update_epoch}'
        plot_loss_name = plot_loss_name + f'_update_{n_update_epoch}'

    # Create a folder for sequence plots if it does not exist
    if not os.path.exists(f'logs/{stock_name}/{date_info}/{time}/seq_{s}/plots'):
        os.makedirs(f'logs/{stock_name}/{date_info}/{time}/seq_{s}/plots')

    plt.clf()

    # Plot and save training and validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'logs/{stock_name}/{date_info}/{time}/seq_{s}/plots/{plot_acc_name}.jpg')

    plt.clf()

    # Plot and save training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'logs/{stock_name}/{date_info}/{time}/seq_{s}/plots/{plot_loss_name}.jpg')
