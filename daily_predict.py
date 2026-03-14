import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model


DATA_PATH = "data/nifty_ohlc.csv"
MODEL_PATH = "model/lstm_attention.h5"
SCALER_PATH = "model/scaler.pkl"

os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

def load_historical_data():

    try:
        data = pd.read_csv(DATA_PATH)

        # Fix if Date column missing
        if "Date" not in data.columns:
            data.reset_index(inplace=True)
            data.rename(columns={"index": "Date"}, inplace=True)

        data["Date"] = pd.to_datetime(data["Date"])

        print("Historical data loaded.")

    except FileNotFoundError:
        print("No historical file found. Downloading full history...")
        data = download_full_history()

    return data

def download_full_history():

    nifty = yf.download("^NSEI", period="max", interval="1d")

    nifty = nifty[['Open','High','Low','Close']]
    nifty.reset_index(inplace=True)

    nifty.to_csv(DATA_PATH, index=False)

    print("Full history downloaded.")

    return nifty

def update_dataset(data):

    last_date = data['Date'].max()

    today = datetime.today()

    print(f"Last stored date: {last_date.date()}")

    new_data = yf.download(
        "^NSEI",
        start=last_date + timedelta(days=1),
        end=today + timedelta(days=1),
        interval="1d"
    )

    if new_data.empty:
        print("No new data available.")
        return data

    new_data = new_data[['Open','High','Low','Close']]
    new_data.reset_index(inplace=True)

    updated = pd.concat([data, new_data])

    updated.drop_duplicates(subset="Date", inplace=True)
    updated.sort_values("Date", inplace=True)

    updated.to_csv(DATA_PATH, index=False)

    print(f"Dataset updated. Added {len(new_data)} rows.")

    return updated

import tensorflow as tf
from tensorflow.keras.layers import Layer

import tensorflow as tf
from tensorflow.keras.layers import Layer

class AttentionLayer(Layer):

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):

        e = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)

        a = tf.nn.softmax(e, axis=1)

        output = inputs * a

        return tf.reduce_sum(output, axis=1)

def load_artifacts():

    model = load_model(
    MODEL_PATH,
    custom_objects={"AttentionLayer": AttentionLayer},
    compile=False
    )
    scaler = joblib.load(SCALER_PATH)

    print("Model & scaler loaded.")

    return model, scaler



def prepare_input_window(data, scaler):

    ohlc = data[['Open','High','Low','Close']]

    # Convert to numeric just in case
    ohlc = ohlc.apply(pd.to_numeric, errors='coerce')
    ohlc = ohlc.dropna()

    if len(ohlc) < 30:
        raise ValueError("Not enough data for 30-day window.")

    last_30 = ohlc.tail(30)

    # Log transform
    log_data = np.log(last_30)

    # Standardize
    scaled_data = scaler.transform(log_data)

    # Reshape for LSTM
    X_input = scaled_data.reshape(1,30,4)

    return X_input


def predict_next_close(model, scaler, X_input):

    pred_scaled_log = model.predict(X_input)

    close_mean = scaler.mean_[3]
    close_std = scaler.scale_[3]

    pred_log = (pred_scaled_log[0][0] * close_std) + close_mean

    pred_close = np.exp(pred_log)

    return pred_close


def save_prediction(pred_close):

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    output_path = "data/daily_predictions.csv"

    new_row = pd.DataFrame({
        "Date": [now],
        "Predicted_Close": [pred_close]
    })

    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
    else:
        df = pd.DataFrame(columns=["Date","Predicted_Close"])

    df = pd.concat([df,new_row], ignore_index=True)

    df.to_csv(output_path, index=False)

    print("Prediction saved.")



def main():

    print("\n===== NIFTY DAILY PREDICTION =====\n")

    data = load_historical_data()

    data = update_dataset(data)

    model, scaler = load_artifacts()

    X_input = prepare_input_window(data, scaler)

    pred_close = predict_next_close(model, scaler, X_input)

    print(f"\nPredicted Next-Day Close: {pred_close:.2f}")

    save_prediction(pred_close)

    print("\n===== DONE =====\n")


if __name__ == "__main__":
    main()