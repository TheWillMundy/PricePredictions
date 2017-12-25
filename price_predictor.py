import csv
import numpy
from sklearn.svm import SVR
import matplotlib.pyplot as plt

data_dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        # For Stocks 
        next(csvFileReader)
        counter = 0
        for row in csvFileReader:
            data_dates.append(counter)
            # For Stocks
            prices.append(float(row[1]))
            # For Crypto
            # prices.append(float(row[4]))
            counter += 1
    data_dates.reverse()
    return

def predict_price(dates, prices, x, crypto_name):
    # Setup Training & Testing 
    # training_dates = data_dates[:len(data_dates)//2]
    # testing_dates = data_dates[len(data_dates)//2:]
    # training_prices = prices[:len(prices)//2]
    # testing_prices = prices[len(prices)//2:]
    # Reshape Dates 
    # training_dates = numpy.reshape(training_dates, (len(training_dates), 1))
    # testing_dates = numpy.reshape(testing_dates, (len(testing_dates), 1))
    real_dates = numpy.reshape(data_dates, (len(data_dates), 1))
    # 1 SVR Model
    # Best IOTA Gamma
    # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.001)
    # Best SNAP Gamma
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.01)
    # Fit RBF Model
    svr_rbf.fit(real_dates, prices)
    # Plot data
    plt.scatter(real_dates, prices, color='black', label='Data')
    # Plot SVR
    plt.plot(real_dates, svr_rbf.predict(real_dates), color='red', label='RBF model')
    print "Plotted Models"
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(crypto_name)
    plt.legend()
    plt.show()
    
    return svr_rbf.predict(x)[0]

# IOTA Predictions
# get_data('iota_price.csv')
# iota_prediction = predict_price(data_dates, prices, 149, 'Iota')
# print iota_prediction
# Dash Predictions
# get_data('dash_price.csv')
# dash_prediction = predict_price(data_dates, prices, len(data_dates), 'Dash')
# print dash_prediction
# Snap Predictions
get_data('snap_stock.csv')
snap_predictions = predict_price(data_dates, prices, len(data_dates) + 1, 'SNAP')
print snap_predictions
    