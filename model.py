import pandas as pd
import numpy as np
import json
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import flask
from flask import jsonify


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
    test_data = np.reshape(
        test_data, (test_data.shape[0], test_data.shape[1], 1))

    predictions = model.predict(test_data)
    predictions = scaler.inverse_transform(predictions)

    return {'tic': tic,
            'close': json.dumps(close_prices.values.tolist()),
            'predictions': json.dumps(predictions.tolist())}


@app.route('/stocks/<name>')
def get_predictions(name):
    model = keras.models.load_model('./model/')

    predictions_data = []

    tic_data = predict_stocks(name, model)
    predictions_data.append(json.dumps(tic_data))

    return jsonify(predictions_data)
