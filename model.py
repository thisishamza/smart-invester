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
    dates = data.filter(['date'])
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

    closes=close_prices.values.tolist()
    datas=dates.values.tolist()
    preds=predictions.tolist()

    data = {}

    for x,y,z in zip(closes,datas,preds):
        test = {}
        test['close'] = x[0]
        test['date'] = y[0]
        test['prediction'] = z[0]
        # data.update(test)
        
        if "data" in data:
            data["data"].append(test)
        else:
            data["data"] = [test]

        print(data)

    # for i in range(len(preds)):
    #     test = {}
    #     test['close'] = closes[i]
    #     test['date'] = datas[i]
    #     test['prediction'] = preds[i]
    #     data['data'] = test
    
    return data
    # return {'tic': tic,
    #         'close': (close_prices.values.tolist()),
    #         'predictions': (predictions.tolist()),
    #         'date': (dates.values.tolist())
    #         }


@app.route('/stocks/<name>')
def get_predictions(name):
    model = keras.models.load_model('./model/')
    tic_data = predict_stocks(name, model)

    return tic_data
