import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
import joblib

# Load models and feature selector
lstm_model = load_model("model_lstm.h5")
xgb_model = joblib.load("model_xgb.pkl")
ridge_model = joblib.load("model_ridge.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("feature_selector.pkl")

# Function to create features
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
    for lag in [1, 3, 7]:
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
        data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
    data['Rolling_Mean_7'] = data['Close'].rolling(window=7).mean()
    data['Rolling_Std_7'] = data['Close'].rolling(window=7).std()
    data['Daily_Return'] = data['Close'].pct_change()
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    data.reset_index(inplace=True)
    data.fillna(data.median(), inplace=True)
    return data

# Function to preprocess data for models
def preprocess_data(data, scaler, selector):
    features = [
        'Open', 'High', 'Low', 'Adj Close', 'Volume', 'SMA_7', 'SMA_30', 'EMA_7', 'EMA_30', 'RSI_14',
        'BB_High', 'BB_Low', 'BB_Width', 'ATR', 'Close_Lag_1', 'Close_Lag_3', 'Close_Lag_7',
        'Volume_Lag_1', 'Volume_Lag_3', 'Volume_Lag_7', 'Rolling_Mean_7', 'Rolling_Std_7',
        'Daily_Return', 'Log_Return'
    ]
    X = data[features]
    X_scaled = scaler.transform(X)
    optimal_features = X.columns[selector.support_]
    X_selected = X[optimal_features]
    X_scaled_selected = scaler.transform(X_selected)
    return X_scaled_selected

# Streamlit app
st.title("Crypto Price Prediction")

# Sidebar for user input
ticker = st.sidebar.text_input("Enter Crypto Ticker (e.g., ONDO-USD)", "ONDO-USD")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2024-01-14'))
end_date = st.sidebar.date_input("End Date", pd.to_datetime('today'))

# Download data
try:
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data.reset_index()

    if data.empty:
        st.error("No data found for the given ticker and date range.")
    else:
        # Feature creation and preprocessing
        data = create_features(data)
        X_scaled_selected = preprocess_data(data, scaler, selector)

        # Predictions for the next day only
        # LSTM Prediction
        X_lstm_last = X_scaled_selected[-1].reshape((1, X_scaled_selected.shape[1], 1)) # Reshape only the last day's data
        lstm_pred = lstm_model.predict(X_lstm_last)[0][0]

        # XGBoost Prediction
        xgb_pred = xgb_model.predict(X_scaled_selected[-1].reshape(1, -1))[0] # Predict using only the last day's data

        # Ridge Prediction
        ridge_pred = ridge_model.predict(X_scaled_selected[-1].reshape(1, -1))[0] # Predict using only the last day's data

        # Display predictions
        st.subheader("Predictions for the next day:")
        st.write(f"LSTM Prediction: {lstm_pred:.4f}")
        st.write(f"XGBoost Prediction: {xgb_pred:.4f}")
        st.write(f"Ridge Prediction: {ridge_pred:.4f}")

        # Plot historical data
        st.subheader("Historical Data")
        st.line_chart(data.set_index('Date')['Close'])

except Exception as e:
    st.error(f"An error occurred: {e}")
