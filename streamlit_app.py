import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
from datetime import datetime, timedelta

# Function to fetch data from Yahoo Finance
def fetch_latest_data(ticker):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=6*30)  # Approx. 6 months
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    st.write(f"Fetching data for {ticker} from {start_date_str} to {end_date_str}...")
    data = yf.download(ticker, start=start_date_str, end=end_date_str)
    return data.reset_index()

# Function to create features
def create_features(data):
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

    return data.fillna(data.median())

# Function to load models and scaler
def load_models_and_scaler():
    lstm_model = load_model("model_lstm.h5")
    xgb_model = joblib.load("xgb_model.pkl")
    ridge_model = joblib.load("ridge_model.pkl")
    scaler = joblib.load("scaler.pkl")
    optimal_features = joblib.load("optimal_features.pkl")
    return lstm_model, xgb_model, ridge_model, scaler, optimal_features

# Function to prepare the input for prediction
def prepare_input_for_prediction(data, scaler, optimal_features):
    # Select only optimal features
    X_next_day = data[optimal_features].iloc[-1:]  # Get the last row
    # Scale the features
    X_next_day_scaled = scaler.transform(X_next_day)
    return X_next_day, X_next_day_scaled

# Function to make predictions
def predict_next_day_price(lstm_model, xgb_model, ridge_model, X_next_day_scaled):
    # Prepare data for LSTM model
    X_next_day_lstm = X_next_day_scaled.reshape((X_next_day_scaled.shape[0], X_next_day_scaled.shape[1], 1))

    # Predict using all models
    lstm_prediction = lstm_model.predict(X_next_day_lstm, verbose=0).flatten()[0]  # Ensure scalar
    xgb_prediction = xgb_model.predict(X_next_day_scaled).flatten()[0]  # Ensure scalar
    ridge_prediction = ridge_model.predict(X_next_day_scaled).flatten()[0]  # Ensure scalar

    return {
        "LSTM": lstm_prediction,
        "XGBoost": xgb_prediction,
        "Ridge": ridge_prediction
    }


# Streamlit app
def main():
    st.title("Next Day Price Prediction")

    # Input for the ticker symbol
    ticker = st.text_input("Enter the Ticker Symbol (e.g., AAPL, TSLA, ONDO-USD):", value="AAPL")

    # Button for prediction
    if st.button("Predict Next Day Price"):
        try:
            # Fetch the latest data
            data = fetch_latest_data(ticker)
            data = create_features(data)

            # Load saved models and scaler
            st.write("Loading models and preparing data...")
            lstm_model, xgb_model, ridge_model, scaler, optimal_features = load_models_and_scaler()

            # Prepare input for prediction
            X_next_day, X_next_day_scaled = prepare_input_for_prediction(data, scaler, optimal_features)

            # Make predictions
            predictions = predict_next_day_price(lstm_model, xgb_model, ridge_model, X_next_day_scaled)

            # Display results
            st.write("Next Day Price Predictions:")
            st.write(f"LSTM Prediction: {predictions['LSTM']:.2f}")
            st.write(f"XGBoost Prediction: {predictions['XGBoost']:.2f}")
            st.write(f"Ridge Prediction: {predictions['Ridge']:.2f}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
