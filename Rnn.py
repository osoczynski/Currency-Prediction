import Config
import numpy
from keras.layers import Dense
from keras.models import Sequential

numpy.random.seed(7)


def modify_data_set_rnn(training_set, testing_set):
    train_actual = []
    train_predict = []
    for interval in range(len(training_set) - Config.previous_data_points - 1):
        train_actual.append(training_set[interval: interval + Config.previous_data_points])
        train_predict.append(training_set[interval + Config.previous_data_points])

    test_actual = []
    test_predict = []
    for interval in range(len(testing_set) - Config.previous_data_points - 1):
        test_actual.append(testing_set[interval: interval + Config.previous_data_points])
        test_predict.append(testing_set[interval + Config.previous_data_points])

    return train_actual, train_predict, test_actual, test_predict


def build_recurrent_neural_network(train_actual, train_predict):
    recurrent_neural_network = Sequential()

    recurrent_neural_network.add(Dense(12, input_dim=Config.previous_data_points, activation="relu"))
    recurrent_neural_network.add(Dense(8, activation="relu"))
    recurrent_neural_network.add(Dense(1))

    recurrent_neural_network.compile(loss='mean_squared_error', optimizer='adam')
    recurrent_neural_network.fit(train_actual, train_predict, epochs=Config.epochs, batch_size=Config.batch_size, verbose=0)

    return recurrent_neural_network


def predict_rnn(recurrent_neural_network, train_actual, test_actual):
    training_predict = recurrent_neural_network.predict(train_actual)
    testing_predict = recurrent_neural_network.predict(test_actual)

    testing_predict_rnn = [None] * Config.data_set_length
    testing_predict_rnn[Config.previous_data_points - 1:len(training_predict) + Config.previous_data_points] = \
        list(testing_predict[:, 0])

    return testing_predict_rnn


def rnn_model(training_set, testing_set):
    train_actual, train_predict, test_actual, test_predict = modify_data_set_rnn(training_set, testing_set)
    rnn = build_recurrent_neural_network(train_actual, train_predict)
    testing_predict_rnn = predict_rnn(rnn, train_actual, test_actual)

    return testing_predict_rnn
