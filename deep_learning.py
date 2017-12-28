import pandas as pd 
import time
import seaborn as sns
import matplotlib.pyplot as plt 
import datetime
import numpy as np
# Keras imports
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
# to load existing models
from keras.models import load_model
# For training and testing sets 
from sklearn.model_selection import train_test_split
# For plotting
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# Mins and Maxes necessary for normalization
MINIMUM_VOLUME = 0
MAXIMUM_VOLUME = 0
MINIMUM_CLOSE = 0
MAXIMUM_CLOSE = 0

def calculate_close_off_high(row):
    """
    Normalized Data
    Algorithm: a + (((x - A)(b - a))/(B - A))
    a = -1, b = 1 => (b-a = 2)
    A = Daily High, B = Daily Low
    X = Close Price
    """
    high, low, close = row['High'], row['Low'], row['Close']
    numerator = 2*(close - high)
    denominator = (low - high)
    print low, high
    if denominator == 0:
        return 0
    return (-1) + (numerator / float(denominator))

def calculate_volatility(row):
    """
    Algorithm: (a - b)/c
    a = high, b = low, c = open_price
    """
    high, low, open_price = row['High'], row['Low'], row['Open']
    numerator = high - low 
    if (open_price == 0 or type(open_price) != float):
        return 0
    return numerator/float(open_price)

def fix_volume(row):
    """
    Sets volume equal to 0 if volume = '-'
    """
    if row['Volume'] == '-':
        return 0
    return row['Volume']

def find_min_max(dataframe):
    """
    """
    # Keeping pythonic scope in mind
    global MINIMUM_VOLUME
    global MAXIMUM_VOLUME
    global MINIMUM_CLOSE
    global MAXIMUM_CLOSE
    # Get high and low
    dataframe_aggregations = dataframe.agg({'Volume': ['min', 'max'], 'Close': ['min', 'max']})
    MINIMUM_VOLUME = dataframe_aggregations.iloc[0]['Volume']
    MAXIMUM_VOLUME = dataframe_aggregations.iloc[1]['Volume']
    MINIMUM_CLOSE = dataframe_aggregations.iloc[0]['Close']
    MAXIMUM_CLOSE = dataframe_aggregations.iloc[1]['Close']

def normalize_volume(row):
    """
    Algorithm: a + (((x - A)(b - a))/(B - A))
    a = 1, b = -1 => (b-a = -2)
    A = Highest, B = Lowest 
    X = Current
    """
    numerator = (-2) * (row['Volume'] - MAXIMUM_VOLUME)
    denominator = MINIMUM_VOLUME - MAXIMUM_VOLUME
    if denominator == 0:
        return 0
    return 1 + (numerator / float(denominator))
    
def normalize_close_prices(row):
    """
    Algorithm: a + (((x - A)(b - a))/(B - A))
    a = 1, b = -1 => (b-a = -2)
    A = Highest, B = Lowest 
    X = Current
    """
    numerator = (-2) * (row['Close'] - MAXIMUM_CLOSE)
    denominator = MINIMUM_CLOSE - MAXIMUM_CLOSE
    if denominator == 0:
        return 0
    return 1 + (numerator / float(denominator))

def normalize(dataframe):
    """
    
    """
    normalized_volumes = dataframe.apply(normalize_volume, axis=1)
    normalized_close_prices = dataframe.apply(normalize_close_prices, axis=1)
    return normalized_volumes, normalized_close_prices

def check_training_error(model_history):
    # Plot training error
    fig, ax1 = plt.subplots(1,1)
    ax1.plot(model_history.epoch, model_history.history['loss'])
    ax1.set_title('Training Error')
    if model_history.history['loss'] == 'mae':
        ax1.set_ylabel('Mean Absolute Error (MAE)',fontsize=12)
    # just in case you decided to change the model loss calculation
    else:
        ax1.set_ylabel('Model Loss',fontsize=12)
    ax1.set_xlabel('# Epochs',fontsize=12)
    plt.show()

def check_single_timepoint(initial_data, model, LSTM_testing_inputs, testing_set, window_len, split_date):
    fig, ax1 = plt.subplots(1,1)
    ax1.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
    ax1.set_xticklabels([datetime.date(2017,i+1,1).strftime('%b %d %Y')  for i in range(12)])
    ax1.plot(initial_data[initial_data['Date'] > split_date]['Date'][10:].astype(datetime.datetime),
             testing_set['Close'][window_len:], label='Actual')
    ax1.plot(initial_data[initial_data['Date'] > split_date]['Date'][10:].astype(datetime.datetime),
             ((np.transpose(model.predict(LSTM_testing_inputs))+1) * testing_set['Close'].values[:-window_len])[0], 
             label='Predicted')
    ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(model.predict(LSTM_testing_inputs))+1)-\
                (testing_set['Close'].values[window_len:])/(testing_set['Close'].values[:-window_len]))), 
                 xy=(0.75, 0.9),  xycoords='axes fraction',
                xytext=(0.75, 0.9), textcoords='axes fraction')
    ax1.set_title('Test Set: Single Timepoint Prediction',fontsize=13)
    ax1.set_ylabel('Price ($)',fontsize=12)
    ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
    plt.show()

def get_crypto_data(crypto_name):
    # get market info
    url = "https://coinmarketcap.com/currencies/{crypto}/historical-data/?start=20130428&end=".format(crypto=crypto_name)+time.strftime("%Y%m%d")
    market_info = pd.read_html(url)[0]
    # convert the date string to the correct date format
    market_info = market_info.assign(Date=pd.to_datetime(market_info['Date']))
    # convert high, low, open, close to numbers
    market_info['High'] = market_info['High'].astype('float64')
    market_info['Low'] = market_info['Low'].astype('float64')
    market_info['Open'] = market_info['Open'].astype('float64')
    market_info['Close'] = market_info['Close'].astype('float64')
    # create new series of dataframes
    model_info = market_info[['Date', 'Close', 'Volume']].copy()
    # add new columns
    model_info['Close_Off_High'] = market_info.apply(calculate_close_off_high, axis=1)
    model_info['Volatility'] = market_info.apply(calculate_volatility, axis=1)
    # fix volume
    model_info['Volume'] = model_info.apply(fix_volume, axis=1)
    model_info['Volume'] = model_info['Volume'].astype('int64')
    # normalize data
    find_min_max(model_info)
    model_info['Volume'], model_info['Close'] = normalize(model_info)
    # Reverses order of dataframes, so that oldest are now at head
    # What's going on in longer code: model_info = model_info.reindex(index=model_info.index[::-1])
    model_info = model_info.iloc[::-1]
    # # look at the first few rows
    # print model_info.head()
    # Return dataset
    return model_info
    
def build_model(inputs, output_size, neurons, activ_func = "linear", 
                dropout = 0.25, loss = "mae", optimizer = "adam"):
    # Type of model 
    model = Sequential()
    # Setup LSTM
    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    # Setup Dropout to reduce overfitting
    model.add(Dropout(dropout))
    # Add Dense 
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    # Finish building model 
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

def setup_inputs_outputs(crypto):
    """
    """
    # Get Data
    initial_data = get_crypto_data(crypto)
    # Separate data into training and testing sets 
    training_set, testing_set = train_test_split(initial_data, test_size=0.2, shuffle=False)
    # split date 
    split_date = (training_set.tail(1)).iloc[0]['Date']
    # We no longer need dates
    training_set = training_set.drop('Date', axis=1)
    testing_set = testing_set.drop('Date', axis=1)
    # window_len tells us how many previous datapoints we keep track of
    window_len = 10
    LSTM_training_inputs = [] # Initialize
    LSTM_testing_inputs = []
    # Create training inputs
    for index in range(len(training_set) - window_len):
        temp_set = training_set[index: (index + window_len)].copy()
        # Normalizes each column further to be aligned with rest of window...
        # for col in norm_cols:
        #     temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
        LSTM_training_inputs.append(temp_set)
    # Create testing inputs 
    for index in range(len(testing_set) - window_len):
        temp_set = testing_set[index: (index + window_len)].copy()
        LSTM_testing_inputs.append(temp_set)
    # Getting input dimensions
    # print LSTM_testing_inputs
    # Turn into numpy array 
    LSTM_training_inputs = np.array([np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs])
    LSTM_testing_inputs = np.array([np.array(LSTM_testing_input) for LSTM_testing_input in LSTM_testing_inputs])
    # model output is next price normalised to 10th previous closing price
    LSTM_training_outputs = (training_set['Close'][window_len:].values/training_set['Close'][:-window_len].values)-1
    LSTM_test_outputs = (testing_set['Close'][window_len:].values/testing_set['Close'][:-window_len].values)-1
    return LSTM_training_inputs, LSTM_training_outputs, LSTM_testing_inputs, LSTM_test_outputs, training_set, testing_set, split_date

def save_model(crypto):
    initial_data = get_crypto_data(crypto)
    LSTM_training_inputs, LSTM_training_outputs, LSTM_testing_inputs, LSTM_test_outputs, training_set, testing_set, split_date = setup_inputs_outputs(crypto)
    # # Get Data
    # initial_data = get_crypto_data(crypto)
    # # We no longer need dates
    # crypto_data = initial_data.drop('Date', axis=1)
    # # Separate data into training and testing sets 
    # training_set, testing_set = train_test_split(crypto_data, test_size=0.2, shuffle=False)
    # # window_len tells us how many previous datapoints we keep track of
    # window_len = 10
    # LSTM_training_inputs = [] # Initialize
    # LSTM_testing_inputs = []
    # # Create training inputs
    # for index in range(len(training_set) - window_len):
    #     temp_set = training_set[index: (index + window_len)].copy()
    #     # Normalizes each column further to be aligned with rest of window...
    #     # for col in norm_cols:
    #     #     temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    #     LSTM_training_inputs.append(temp_set)
    # # Create testing inputs 
    # for index in range(len(testing_set) - window_len):
    #     temp_set = testing_set[index: (index + window_len)].copy()
    #     LSTM_testing_inputs.append(temp_set)
    # # Getting input dimensions
    # print LSTM_testing_inputs
    # # Turn into numpy array 
    # LSTM_training_inputs = np.array([np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs])
    # LSTM_testing_inputs = np.array([np.array(LSTM_testing_input) for LSTM_testing_input in LSTM_testing_inputs])
    
    # # Get several days' worth of predictions
    # # random seed for reproducibility
    # np.random.seed(202)
    # # we'll try to predict the closing price for the next 10 days 
    # # change this value if you want to make longer/shorter prediction
    # pred_range = 10
    # # initialise model architecture
    # model = build_model(LSTM_training_inputs, output_size=pred_range, neurons = 20)
    # # model output is next 5 prices normalised to 10th previous closing price
    # LSTM_training_outputs = []
    # for i in range(window_len, len(training_set['Close'])-pred_range):
    #     LSTM_training_outputs.append((training_set['Close'][i:i+pred_range].values/
    #                                   training_set['Close'].values[i-window_len])-1)
    # LSTM_training_outputs = np.array(LSTM_training_outputs)
    # # train model on data
    # # note: eth_history contains information on the training error per epoch
    # model_history = model.fit(LSTM_training_inputs[:-pred_range], LSTM_training_outputs, 
    #                             epochs=50, batch_size=1, verbose=2, shuffle=True)
    
    # random seed for reproducibility
    np.random.seed(202)
    # Initialize model 
    model = build_model(LSTM_training_inputs, output_size=1, neurons=20)
    # # model output is next price normalised to 10th previous closing price
    # LSTM_training_outputs = (training_set['Close'][window_len:].values/training_set['Close'][:-window_len].values)-1
    # LSTM_test_outputs = (testing_set['Close'][window_len:].values/testing_set['Close'][:-window_len].values)-1
    # train model on data
    # note: model_history contains information on the training error per epoch
    model_history = model.fit(LSTM_training_inputs, LSTM_training_outputs, 
                                epochs=50, batch_size=1, verbose=2, shuffle=True)
    
    # save model
    model.save('{}_model.h5'.format(crypto))
    
    # # Check training error 
    # check_training_error(model_history)
    # 
    # # Plot subset of data
    # check_single_timepoint(initial_data, model, LSTM_testing_inputs, testing_set, 10, split_date)
    
    #eth_preds = np.loadtxt('eth_preds.txt')

def load_and_run_model(crypto):
    """
    
    """
    model_name = crypto + "_model.h5"
    model = load_model(model_name)
    LSTM_training_inputs, LSTM_training_outputs, LSTM_testing_inputs, LSTM_test_outputs, training_set, testing_set, split_date = setup_inputs_outputs(crypto)
    scores = model.evaluate(LSTM_training_inputs, LSTM_training_outputs)
    # print "Training: ", scores
    # scores = model.evaluate(LSTM_testing_inputs, LSTM_test_outputs)
    # print "Testing: ", scores
    initial_data = get_crypto_data('bitcoin')
    check_single_timepoint(initial_data, model, LSTM_testing_inputs, testing_set, 10, split_date)
    # # We no longer need dates
    # crypto_data = initial_data.drop('Date', axis=1)
    # # Separate data into training and testing sets 
    # training_set, testing_set = train_test_split(crypto_data, test_size=0.2, shuffle=False)
    # # window_len tells us how many previous datapoints we keep track of
    # window_len = 10
    # LSTM_testing_inputs = []
    # # Create testing inputs 
    # for index in range(len(testing_set) - window_len):
    #     temp_set = testing_set[index: (index + window_len)].copy()
    #     LSTM_testing_inputs.append(temp_set)
    # LSTM_testing_inputs = np.array([np.array(LSTM_testing_input) for LSTM_testing_input in LSTM_testing_inputs])
    # LSTM_test_outputs = (testing_set['Close'][window_len:].values/testing_set['Close'][:-window_len].values)-1
    # scores = model.evaluate(LSTM_testing_inputs, LSTM_test_outputs)
    # print("%s: %.2f%%" % (model.metrics_names, scores[1]*100))
    # print scores

def run_model(crypto):
    """
    
    """
    # first save model 
    save_model(crypto)
    load_and_run_model(crypto)        

run_model('dash')
# print get_crypto_data('iota')