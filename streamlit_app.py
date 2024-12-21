import yfinance as yf
import pandas as pd
import numpy as np
import ta
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
import streamlit as st
import joblib
import plotly.graph_objects as go
import os

# Function to fetch data
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data.reset_index()
    return data

# Function to create features (updated for robustness)
def create_features(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Correctly create new columns from indicators
    data['SMA_7'] = SMAIndicator(close=data['Close'], window=7).sma_indicator()
    data['SMA_30'] = SMAIndicator(close=data['Close'], window=30).sma_indicator()
    data['EMA_7'] = EMAIndicator(close=data['Close'], window=7).ema_indicator()
    data['EMA_30'] = EMAIndicator(close=data['Close'], window=30).ema_indicator()
    data['RSI_14'] = RSIIndicator(close=data['Close'], window=14).rsi()

    bb_indicator = BollingerBands(close=data['Close'], window=20, window_dev=2)
    data['BB_High'] = bb_indicator.bollinger_hband()
    data['BB_Low'] = bb_indicator.bollinger_lband()
    data['BB_Width'] = data['BB_High'] - data['BB_Low']

    data['ATR'] = ta.volatility.average_true_range(high=data['High'], low=data['Low'], close=data['Close'], window=14)

    lags = [1, 3, 7]
    for lag in lags:
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
        data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)

    data['Rolling_Mean_7'] = data['Close'].rolling(window=7).mean()
    data['Rolling_Std_7'] = data['Close'].rolling(window=7).std()
    data['Daily_Return'] = data['Close'].pct_change()
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))

    # Fill NaN values appropriately
    data.fillna(method='bfill', inplace=True)  # Use bfill to fill indicator NaNs
    data.fillna(0, inplace=True)  # Fill remaining NaNs with 0

    data.reset_index(inplace=True)
    return data

# Function for next-day prediction
def predict_next_day(X_next_day, lstm_model, xgb_model, ridge_model, scaler):
    X_next_day_scaled = scaler.transform(X_next_day)
    
    # Reshape data for LSTM
    X_next_day_lstm = X_next_day_scaled.reshape((X_next_day_scaled.shape[0], X_next_day_scaled.shape[1], 1))
    
    # Mitigate the error for Ridge and XGBoost
    xgb_next_day_prediction = xgb_model.predict(X_next_day_scaled)[0]
    ridge_next_day_prediction = ridge_model.predict(X_next_day_scaled)[0]
    
    # Predict
    lstm_next_day_prediction = lstm_model.predict(X_next_day_lstm, verbose=0)[0][0]

    return lstm_next_day_prediction, xgb_next_day_prediction, ridge_next_day_prediction

def main():
    st.title("Stock Price Prediction (Next Day)")

    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., ONDO-USD)", "ONDO-USD")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-01-14"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-18"))

    if st.sidebar.button("Fetch Data"):
        data = fetch_data(ticker, start_date, end_date)
        st.write("Data Preview:", data.head())

        with st.spinner('Creating features...'):
            data = create_features(data)
            st.success('Features created!')

        if st.sidebar.button("Predict Next Day's Price"):
            try:
                # Load the saved models and scaler
                lstm_model = load_model("lstm_model.pkl")
                xgb_model = joblib.load("xgb_model.pkl")
                ridge_model = joblib.load("ridge_model.pkl")
                scaler = joblib.load("scaler.pkl")
                optimal_features = joblib.load("optimal_features.pkl")

                with st.spinner('Predicting...'):
