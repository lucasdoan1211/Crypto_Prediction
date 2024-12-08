import streamlit as st
import pandas as pd
import numpy as np
import ta  
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
import pickle
import yfinance as yf
from datetime import datetime, timedelta
import os

# Load pre-trained models and scaler
@st.cache_resource
def load_artifacts():
    try:
        lstm_model = load_model("model_lstm.h5")
        xgb_model = pickle.load(open("model_xgb.pkl", "rb"))
        ridge_model = pickle.load(open("model_ridge.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return lstm_model, xgb_model, ridge_model, scaler
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.stop()

lstm_model, xgb_model, ridge_model, scaler = load_artifacts()

# Streamlit UI
st.title("Next-Day Prediction Deployment with Feature Engineering")
st.sidebar.header("Data Source")

# Input for ticker symbol with fixed 1-year range
ticker = st.sidebar.text_input("Enter Ticker Symbol", "AAPL")
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

if st.sidebar.button("Fetch Data, Engineer Features, and Predict Next Day"):
    try:
        # Fetch data from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date)
        st.write(f"### Data for {ticker} (Last 1 Year)")
        st.write(data.tail())

        if data.empty:
            st.error("No data fetched. Please check the ticker symbol or try again later.")
            st.stop()

        # Feature Engineering
        data['Date'] = data.index
        data.reset_index(drop=True, inplace=True)

        # 1. Moving Averages (SMA and EMA)
        data['SMA_7'] = SMAIndicator(close=data['Close'], window=7).sma_indicator()
        data['SMA_30'] = SMAIndicator(close=data['Close'], window=30).sma_indicator()
        data['EMA_7'] = EMAIndicator(close=data['Close'], window=7).ema_indicator()
        data['EMA_30'] = EMAIndicator(close=data['Close'], window=30).ema_indicator()

        # 2. Relative Strength Index (RSI)
        data['RSI_14'] = RSIIndicator(close=data['Close'], window=14).rsi()

        # 3. Bollinger Bands
        bb_indicator = BollingerBands(close=data['Close'], window=20, window_dev=2)
        data['BB_High'] = bb_indicator.bollinger_hband()
        data['BB_Low'] = bb_indicator.bollinger_lband()
        data['BB_Width'] = data['BB_High'] - data['BB_Low']

        # 4. Average True Range (ATR) - Volatility
        data['ATR'] = ta.volatility.average_true_range(
            high=data['High'], low=data['Low'], close=data['Close'], window=14
        )

        # 5. Lag Features
        lags = [1, 3, 7]
        for lag in lags:
            data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
            data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)

        # 6. Rolling Statistics
        data['Rolling_Mean_7'] = data['Close'].rolling(window=7).mean()
        data['Rolling_Std_7'] = data['Close'].rolling(window=7).std()

        # 7. Returns
        data['Daily_Return'] = data['Close'].pct_change()
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))

        data.dropna(inplace=True)
        st.write("### Feature-Engineered Data")
        st.write(data.tail())

        # Preprocessing
        try:
            data_scaled = scaler.transform(data.drop(columns=['Date'], errors='ignore'))
            X_lstm = data_scaled.reshape((data_scaled.shape[0], data_scaled.shape[1], 1))
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")
            st.stop()

        # Predictions
        st.write("### Next-Day Predictions")
        next_day_data = data_scaled[-1].reshape(1, -1)
        next_day_lstm = X_lstm[-1].reshape(1, X_lstm.shape[1], 1)

        ridge_prediction = ridge_model.predict(next_day_data)[0]
        xgb_prediction = xgb_model.predict(next_day_data)[0]
        lstm_prediction = lstm_model.predict(next_day_lstm).flatten()[0]

        # Combine results
        predictions = {
            "Ridge Prediction": ridge_prediction,
            "XGBoost Prediction": xgb_prediction,
            "LSTM Prediction": lstm_prediction
        }
        st.write(predictions)

    except Exception as e:
        st.error(f"Error fetching data or predicting: {e}")
else:
    st.write("Click the button to fetch data, engineer features, and predict the next day.")
