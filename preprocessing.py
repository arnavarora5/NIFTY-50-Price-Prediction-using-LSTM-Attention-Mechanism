import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler


os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)



data = pd.read_csv("data/nifty_ohlc.csv")


if "Date" not in data.columns:
    data.reset_index(inplace=True)
    data.rename(columns={"index": "Date"}, inplace=True)


ohlc = data[['Open','High','Low','Close']]


ohlc = ohlc.apply(pd.to_numeric, errors='coerce')

ohlc = ohlc.dropna()

print("Original Data Shape:", ohlc.shape)



ohlc = ohlc[(ohlc > 0).all(axis=1)]

log_data = np.log(ohlc)

print("Log transformation applied")



scaler = StandardScaler()

scaled_data = scaler.fit_transform(log_data)

print("Standardization completed")


processed_data = pd.DataFrame(
    scaled_data,
    columns=['Open','High','Low','Close']
)

processed_data["Date"] = data.loc[processed_data.index, "Date"]



processed_data.to_csv("data/nifty_processed.csv", index=False)

print("Processed dataset saved")


joblib.dump(scaler, "model/scaler.pkl")

print("Scaler saved")