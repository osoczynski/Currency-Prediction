import warnings
import matplotlib
from matplotlib import pyplot
from pandas import read_csv
from Rnn import *
from Arima import *


def load_data():
    data_set_frame = read_csv('exchange_rates.csv', header=0,
                              index_col=0, squeeze=True)
    column_headers = data_set_frame.columns.values.tolist()
    Config.currency_index = column_headers.index(Config.currency.upper() + '/PLN') + 1
    data_file = read_csv("exchange_rates.csv", usecols=[Config.currency_index], engine='python')
    raw_data = []  #  convert a matrix values to a simple list of values
    for data_point in data_file.values.tolist():
        raw_data.append(data_point[0])
    Config.data_set_length = len(raw_data)
    return raw_data


def split_data(raw_data):
    Config.training_set_length = int(Config.data_set_length * Config.training_percent)
    Config.testing_set_length = Config.data_set_length - Config.training_set_length
    training_set = raw_data[0:Config.training_set_length]
    testing_set = raw_data[Config.training_set_length:Config.data_set_length]
    return training_set, testing_set


def plot(raw_data, testing_predict_rnn, testing_predict_arima, ):
    actual = pyplot.plot(raw_data[int(Config.training_percent * Config.data_set_length):], label="real value",
                         color="green")
    testing_rnn = pyplot.plot(testing_predict_rnn, label="RNN", color="red")
    testing_arima = pyplot.plot(testing_predict_arima, label="ARIMA", color="blue")

    pyplot.ylabel('exchange rates ' + Config.currency)
    pyplot.xlabel('number of days')
    pyplot.title(Config.currency + '/PLN' )

    pyplot.legend()
    pyplot.show()


def start():
    warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                            FutureWarning)
    warnings.filterwarnings("ignore")
    print('loading the dataset')
    raw_data = load_data()
    print('splitting training and testing set')
    training_set, testing_set = split_data(raw_data)
    print('building and training model')
    testing_predict_arima = arima_model(training_set, testing_set)
    testing_predict_rnn = rnn_model(training_set, testing_set)
    print('plotting')
    plot(raw_data, testing_predict_rnn, testing_predict_arima)
