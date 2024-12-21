import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import yfinance as yf
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator

# Load models and scaler
@st.cache_resource
def load_saved_models():
    lstm_model = load_model("model_lstm.h5")
    xgb_model = joblib.load("xgb_model.pkl")
    ridge_model = joblib.load("ridge_model.pkl")
    scaler = joblib.load("scaler.pkl")
    optimal_features = joblib.load("optimal_features.pkl")
    return lstm_model, xgb_model, ridge_model, scaler, optimal_features

# Fetch data
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data.reset_index()
    return data

# Create features
def create_features(data):
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

    data['ATR'] = ta.volatility.average_true_range(high=data['High'], low=data['Low'], close=data['Close'], window=14)

    lags = [1, 3, 7]
    for lag in lags:
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
        data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)

    data['Rolling_Mean_7'] = data['Close'].rolling(window=7).mean()
    data['Rolling_Std_7'] = data['Close'].rolling(window=7).std()

    data['Daily_Return'] = data['Close'].pct_change()
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))

    data.reset_index(inplace=True)
    data.fillna(data.median(), inplace=True)
    return data

# Streamlit app
def main():
    st.title("Next-Day Close Price Prediction")

    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, ONDO-USD):", value="ONDO-USD")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=6*30)  # Approx. 6 months

    if st.button("Predict Next Day Price"):
        st.write("Fetching data and making predictions... Please wait.")
        try:
            # Load models and scaler
            lstm_model, xgb_model, ridge_model, scaler, optimal_features = load_saved_models()

            # Fetch and process data
            data = fetch_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            data = create_features(data)

            # Prepare features for prediction
            X_next_day = data[optimal_features].iloc[[-1]]
            X_next_day_scaled = scaler.transform(X_next_day)
            X_next_day_lstm = X_next_day_scaled.reshape((X_next_day_scaled.shape[0], X_next_day_scaled.shape[1], 1))

            # Predict using models
            lstm_prediction = lstm_model.predict(X_next_day_lstm, verbose=0).flatten()[0]
            xgb_prediction = xgb_model.predict(X_next_day_scaled).flatten()[0]
            ridge_prediction = ridge_model.predict(X_next_day_scaled).flatten()[0]

            st.write("### Predictions for Next Day Closing Price:")
            st.write(f"**LSTM Prediction:** {lstm_prediction:.2f}")
            st.write(f"**XGBoost Prediction:** {xgb_prediction:.2f}")
            st.write(f"**Ridge Prediction:** {ridge_prediction:.2f}")

        except Exception as e:
            if "mse" in str(e):
                st.error("An error occurred with the loss function. Ensure the model uses a valid loss function registered with Keras.")
            else:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
