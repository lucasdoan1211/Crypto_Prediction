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
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    return data.reset_index()

# Function to create features
def create_features(data):
    # Feature creation logic
    data['SMA_7'] = SMAIndicator(close=data['Close'], window=7).sma_indicator()
    data['SMA_30'] = SMAIndicator(close=data['Close'], window=30).sma_indicator()
    # Add the rest of your feature engineering here
    return data.fillna(data.median())

# Function to load models and scaler
def load_models_and_scaler():
    lstm_model = load_model("model_lstm.h5")
    xgb_model = joblib.load("xgb_model.pkl")
    ridge_model = joblib.load("ridge_model.pkl")
    scaler = joblib.load("scaler.pkl")
    optimal_features = joblib.load("optimal_features.pkl")
    return lstm_model, xgb_model, ridge_model, scaler, optimal_features

# Function to prepare input for prediction
def prepare_input(data, scaler, optimal_features):
    # Select optimal features and scale
    data = data[optimal_features]
    scaled_data = scaler.transform(data.iloc[-1:])
    return scaled_data

# Function to make predictions
def predict_next_day(lstm_model, xgb_model, ridge_model, scaled_data):
    lstm_input = scaled_data.reshape((1, scaled_data.shape[1], 1))  # Reshape for LSTM
    lstm_prediction = lstm_model.predict(lstm_input, verbose=0).flatten()[0]
    xgb_prediction = xgb_model.predict(scaled_data).flatten()[0]
    ridge_prediction = ridge_model.predict(scaled_data).flatten()[0]
    return {"LSTM": lstm_prediction, "XGBoost": xgb_prediction, "Ridge": ridge_prediction}

# Streamlit app
def main():
    st.title("Next Day Price Prediction")

    # Ticker input
    ticker = st.text_input("Enter Ticker:", value="AAPL")

    if st.button("Predict Next Day Price"):
        data = fetch_latest_data(ticker)
        data = create_features(data)

        lstm_model, xgb_model, ridge_model, scaler, optimal_features = load_models_and_scaler()
        scaled_data = prepare_input(data, scaler, optimal_features)
        predictions = predict_next_day(lstm_model, xgb_model, ridge_model, scaled_data)

        st.write(f"LSTM Prediction: {predictions['LSTM']:.2f}")
        st.write(f"XGBoost Prediction: {predictions['XGBoost']:.2f}")
        st.write(f"Ridge Prediction: {predictions['Ridge']:.2f}")

if __name__ == "__main__":
    main()
