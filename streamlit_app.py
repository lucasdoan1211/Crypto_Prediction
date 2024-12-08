import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler

# Load Models
ridge_model = joblib.load("model_ridge.pkl")
xgb_model = joblib.load("model_xgb.pkl")
lstm_model = load_model("model_lstm.h5")

# Load Scaler
scaler = joblib.load("scaler.pkl")

# Streamlit App Title
st.title("Crypto Price Prediction")
st.write("This app predicts the next day's closing price for a given cryptocurrency using Ridge Regression, XGBoost, and LSTM models.")

# Input Section
ticker = st.text_input("Enter the Cryptocurrency Ticker (e.g., ONDO-USD):", value="BTC-USD")

if st.button("Predict"):
    try:
        # Fetch Data from Yahoo Finance
        data = yf.download(ticker, period="60d", interval="1d")
        data.reset_index(inplace=True)

        if data.empty:
            st.error("No data found for the given ticker. Please try another ticker.")
        else:
            st.write("Data loaded successfully!")

            # Feature Engineering
            data['SMA_7'] = data['Close'].rolling(window=7).mean()
            data['SMA_30'] = data['Close'].rolling(window=30).mean()
            data['EMA_7'] = data['Close'].ewm(span=7, adjust=False).mean()
            data['EMA_30'] = data['Close'].ewm(span=30, adjust=False).mean()
            data['RSI_14'] = (100 - (100 / (1 + data['Close'].pct_change(1).fillna(0).apply(lambda x: max(x, 0)).rolling(window=14).mean() /
                                           data['Close'].pct_change(1).fillna(0).apply(lambda x: abs(x)).rolling(window=14).mean())))
            data['BB_High'] = data['Close'].rolling(window=20).mean() + 2 * data['Close'].rolling(window=20).std()
            data['BB_Low'] = data['Close'].rolling(window=20).mean() - 2 * data['Close'].rolling(window=20).std()
            data['ATR'] = data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min()

            # Create Lag Features
            lags = [1, 3, 7]
            for lag in lags:
                data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
                data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)

            # Fill NaN Values
            data.fillna(method="bfill", inplace=True)

            # Select Features for Prediction
            features = [
                'Open', 'High', 'Low', 'Adj Close', 'Volume', 'SMA_7', 'SMA_30', 'EMA_7', 'EMA_30',
                'RSI_14', 'BB_High', 'BB_Low', 'ATR', 'Close_Lag_1', 'Close_Lag_3', 'Close_Lag_7',
                'Volume_Lag_1', 'Volume_Lag_3', 'Volume_Lag_7'
            ]
            latest_data = data[features].iloc[-1].values.reshape(1, -1)

            # Scale Features
            scaled_data = scaler.transform(latest_data)

            # Ridge Prediction
            ridge_prediction = ridge_model.predict(scaled_data)[0]

            # XGBoost Prediction
            xgb_prediction = xgb_model.predict(scaled_data)[0]

            # LSTM Prediction
            lstm_data = scaled_data.reshape((scaled_data.shape[0], scaled_data.shape[1], 1))
            lstm_prediction = lstm_model.predict(lstm_data).flatten()[0]

            # Display Results
            st.subheader("Predictions for Next Day Closing Price:")
            st.write(f"**Ridge Regression Prediction:** ${ridge_prediction:.2f}")
            st.write(f"**XGBoost Prediction:** ${xgb_prediction:.2f}")
            st.write(f"**LSTM Prediction:** ${lstm_prediction:.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
