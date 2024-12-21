import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator

# Load models and artifacts
scaler = joblib.load("scaler.pkl")
selector = joblib.load("feature_selector.pkl")
xgb_model = joblib.load("model_xgb.pkl")
lstm_model = load_model("model_lstm.h5")
ridge_model = joblib.load("model_ridge.pkl")

# Streamlit app
st.title("Stock Price Prediction App")

# User input for ticker symbol
ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL):", value="AAPL")

# Fetch data
def get_data(ticker):
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(months=6)
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    return data

if st.button("Predict Next Day Close"):
    try:
        # Fetch data
        data = get_data(ticker)
        st.write("### Latest Data")
        st.write(data.tail())

        # Feature creation
        data['SMA_7'] = SMAIndicator(close=data['Close'], window=7).sma_indicator()
        data['SMA_30'] = SMAIndicator(close=data['Close'], window=30).sma_indicator()
        data['EMA_7'] = EMAIndicator(close=data['Close'], window=7).ema_indicator()
        data['EMA_30'] = EMAIndicator(close=data['Close'], window=30).ema_indicator()
        data['RSI_14'] = RSIIndicator(close=data['Close'], window=14).rsi()
        bb_indicator = BollingerBands(close=data['Close'], window=20, window_dev=2)
        data['BB_High'] = bb_indicator.bollinger_hband()
        data['BB_Low'] = bb_indicator.bollinger_lband()
        data['BB_Width'] = data['BB_High'] - data['BB_Low']
        data['ATR'] = (data['High'] - data['Low']).rolling(window=14).mean()
        for lag in [1, 3, 7]:
            data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
            data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
        data['Rolling_Mean_7'] = data['Close'].rolling(window=7).mean()
        data['Rolling_Std_7'] = data['Close'].rolling(window=7).std()
        data['Daily_Return'] = data['Close'].pct_change()
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        data.dropna(inplace=True)

        # Select the latest data for prediction
        latest_data = data.iloc[-1:]
        features = [
            'Open', 'High', 'Low', 'Adj Close', 'Volume', 'SMA_7', 'SMA_30', 'EMA_7', 'EMA_30', 'RSI_14',
            'BB_High', 'BB_Low', 'BB_Width', 'ATR', 'Close_Lag_1', 'Close_Lag_3', 'Close_Lag_7',
            'Volume_Lag_1', 'Volume_Lag_3', 'Volume_Lag_7', 'Rolling_Mean_7', 'Rolling_Std_7',
            'Daily_Return', 'Log_Return'
        ]
        X = latest_data[features]

        # Preprocess features
        X_scaled = scaler.transform(X)
        X_selected = X_scaled[:, selector.support_]

        # Predict with XGBoost (2D input)
        xgb_prediction = xgb_model.predict(X_selected)[0]

        # Predict with LSTM (reshape for 3D input)
        X_lstm = X_selected.reshape((1, X_selected.shape[1], 1))
        lstm_prediction = lstm_model.predict(X_lstm)[0, 0]

        # Predict with Ridge (flatten for 1D input)
        ridge_prediction = ridge_model.predict(X_selected.flatten())[0] # FIX: Flatten the array here

        # Display predictions
        st.write(f"### Predictions for {ticker}")
        st.write(f"- XGBoost Prediction: {xgb_prediction:.2f}")
        st.write(f"- LSTM Prediction: {lstm_prediction:.2f}")
        st.write(f"- Ridge Prediction: {ridge_prediction:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
