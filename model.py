# import os
# import csv
# import time

import pandas as pd
import numpy as np
from flask import Response
from tensorflow import keras
import json
# from selenium import webdriver
# from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import flask
from flask import jsonify
# Create the application.
app = flask.Flask(__name__)


def predict_stocks(tic, model):

    test_data = []
    data = pd.read_csv(f'{tic}.csv')
    close_prices = data.filter(['close'])
    data_set = close_prices.values
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(data_set)
    for i in range(60, len(scaled_data)):
        test_data.append(scaled_data[i-60:i, 0])
    test_data = np.array(test_data)
    test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))

    predictions = model.predict(test_data)
    predictions = scaler.inverse_transform(predictions)

    return {'tic': tic,
            'close': json.dumps(close_prices.values.tolist()),
            'predictions': json.dumps(predictions.tolist())}

@app.route('/stocks')
def get_predictions():
    model = keras.models.load_model('./model/')

    tic_list = ['BOP']
    predictions_data = []

    for tic in tic_list:
        tic_data = predict_stocks(tic, model)
        # tic_data = tic_data.to_ json(orient='columns')
        predictions_data.append(json.dumps(tic_data))

    # data = predictions_data.to_json(orient='columns')
    # predictions_data = json.dumps(predictions_data, indent=4)
    return jsonify(predictions_data)



# pred_data = get_predictions()
# print(pred_data)
# tic_list = ['HBL', 'ENGRO', 'LUCK', 'UBL']
# for tic in tic_list:
#     get_new_data(tic)
#
# data = pd.read_csv('UBL.csv')
#
# close_prices = data.filter(['close'])
#
# data_set = close_prices.values
#
# training_data_len = int(np.ceil(len(data_set) * .95))
#
# from sklearn.preprocessing import MinMaxScaler
#
# scaler = MinMaxScaler(feature_range=(0, 1))
#
# scaled_data = scaler.fit_transform(data_set)
#
# train_data = scaled_data[0:int(training_data_len), :]
#
# x_train = []
#
# y_train = []
#
# for i in range(60, len(train_data)):
#     x_train.append(train_data[i-60:i, 0])
#     y_train.append(train_data[i, 0])
#
# x_train, y_train = np.array(x_train), np.array(y_train)
#
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#
# from keras.models import Sequential
# from keras.layers import Dense, LSTM
#
# # Build the LSTM model
# model = Sequential()
# model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
# model.add(LSTM(64, return_sequences=False))
# model.add(Dense(25))
# model.add(Dense(1))
#
# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')
#
# # Train the model
# model.fit(x_train, y_train, batch_size=5, epochs=20)
#
# # Create the testing data set
# # Create a new array containing scaled values from index 1543 to 2002
# test_data = scaled_data[training_data_len - 60:, :]
# # Create the data sets x_test and y_test
# x_test = []
# y_test = data_set[training_data_len:, :]
# for i in range(60, len(test_data)):
#     x_test.append(test_data[i - 60:i, 0])
#
# # Convert the data to a numpy array
# x_test = np.array(x_test)
#
# # Reshape the data
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
#
# # Get the models predicted price values
# model.save('./model/')
# time.sleep(5)
# model = keras.models.load_model('./model/')
# predictions = model.predict(x_test)
# predictions = scaler.inverse_transform(predictions)
#
# # Get the root mean squared error (RMSE)
# rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
#
# print(rmse)

