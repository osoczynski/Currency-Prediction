import Config
from statsmodels.tsa.arima_model import ARIMA


def build_model_predict_arima(training_set, testing_set):
    testing_predict = list()
    training_predict = list(training_set)

    for testing_set_index in range(Config.testing_set_length):
        arima = ARIMA(training_predict, order=(Config.p, Config.d, Config.q))
        arima_model = arima.fit(disp=0)
        forecasting = arima_model.forecast()[0].tolist()[0]
        testing_predict.append(forecasting)
        training_predict.append(testing_set[testing_set_index])

    return testing_predict

def arima_model(training_set, testing_set):
    testing_predict_arima = build_model_predict_arima(training_set, testing_set)

    return testing_predict_arima
