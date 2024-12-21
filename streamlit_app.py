import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from tensorflow.keras.models import load_model
from xgboost import XGBRegressor
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
import ta

# Load models and scaler
scaler = joblib.load("scaler.pkl")
selector = joblib.load("feature_selector.pkl")
xgb_model = joblib.load("model_xgb.pkl")
ridge_model = joblib.load("model_ridge.pkl")
lstm_model = load_model("model_lstm.h5")

# Title and description
st.title("Cryptocurrency Price Prediction")
st.write("This app predicts the closing price of cryptocurrencies using machine learning models.")

# User input for ticker symbol
ticker = st.text_input("Enter Cryptocurrency Ticker (e.g., ONDO-USD):", "ONDO-USD")

# Fetch new data
if st.button("Predict"):
    with st.spinner("Fetching data and making predictions..."):
        # Fetch data
        data = yf.download(ticker, start="2024-01-14", end="2024-12-18")
        if data.empty:
            st.error("No data found for the provided ticker. Please check the symbol and try again.")
        else:
            # Ensure 'data' is a DataFrame and reset index
            data = pd.DataFrame(data).reset_index()

            # Feature engineering
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            data['SMA_7'] = SMAIndicator(close=data['Close'], window=7).sma_indicator()
            data['SMA_30'] = SMAIndicator(close=data['Close'], window=30).sma_indicator()
            data['EMA_7'] = EMAIndicator(close=data['Close'], window=7).ema_indicator()
            data['EMA_30'] = EMAIndicator(close=data['Close'], window=30).ema_indicator()
            data['RSI_14'] = RSIIndicator(close=data['Close'], window=14).rsi()
            bb_indicator = BollingerBands(close=data['Close'], window=20, window_dev=2)
            data['BB_High'] = bb_indicator.bollinger_hband()
            data['BB_Low'] = bb_indicator.bollinger_lband()
            data['BB_Width'] = data['BB_High'] - data['BB_Low']
            data['ATR'] = ta.volatility.average_true_range(
                high=data['High'], low=data['Low'], close=data['Close'], window=14)
            for lag in [1, 3, 7]:
                data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
                data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
            data['Rolling_Mean_7'] = data['Close'].rolling(window=7).mean()
            data['Rolling_Std_7'] = data['Close'].rolling(window=7).std()
            data['Daily_Return'] = data['Close'].pct_change()
            data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
            data.reset_index(inplace=True)
            data.fillna(data.median(), inplace=True)

            # Feature selection
            features = [
                'Open', 'High', 'Low', 'Adj Close', 'Volume', 'SMA_7', 'SMA_30', 'EMA_7', 'EMA_30', 'RSI_14',
                'BB_High', 'BB_Low', 'BB_Width', 'ATR', 'Close_Lag_1', 'Close_Lag_3', 'Close_Lag_7',
                'Volume_Lag_1', 'Volume_Lag_3', 'Volume_Lag_7', 'Rolling_Mean_7', 'Rolling_Std_7',
                'Daily_Return', 'Log_Return'
            ]
            X = data[features]
            X_scaled = scaler.transform(X)
            X_selected = X_scaled[:, selector.support_]

            # XGBoost Prediction
            xgb_pred = xgb_model.predict(X_selected)

            # Ridge Regression Prediction
            ridge_pred = ridge_model.predict(X_selected)

            # LSTM Prediction
            X_lstm = X_selected.reshape((X_selected.shape[0], X_selected.shape[1], 1))
            lstm_pred = lstm_model.predict(X_lstm)

            # Display predictions
            st.subheader("Predictions")
            st.write(f"XGBoost Prediction: {xgb_pred[-1]:.2f}")
            st.write(f"Ridge Regression Prediction: {ridge_pred[-1]:.2f}")
            st.write(f"LSTM Prediction: {lstm_pred[-1][0]:.2f}")
