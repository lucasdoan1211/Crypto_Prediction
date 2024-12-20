import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# Load models and scaler
scaler = joblib.load("scaler.pkl")
feature_selector = joblib.load("feature_selector.pkl")
xgb_model = joblib.load("model_xgb.pkl")
ridge_model = joblib.load("model_ridge.pkl")
lstm_model = load_model("model_lstm.h5")

# Streamlit App
st.title("Stock Price Prediction")
st.write("This app predicts the next day's stock price based on historical data.")

# User input for stock ticker
ticker = st.text_input("Enter a stock ticker (e.g., ONDO-USD):", "ONDO-USD")

if st.button("Predict Next Day Price"):
    # Fetch last 6 months of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=6 * 30)  # Approx. 6 months
    data = yf.download(ticker, start=start_date, end=end_date)

    if not data.empty:
        # Feature engineering
        data['SMA_7'] = data['Close'].rolling(window=7).mean()
        data['SMA_30'] = data['Close'].rolling(window=30).mean()
        data['EMA_7'] = data['Close'].ewm(span=7, adjust=False).mean()
        data['EMA_30'] = data['Close'].ewm(span=30, adjust=False).mean()
        data['RSI_14'] = (100 - (100 / (1 + data['Close'].pct_change().rolling(window=14).mean() /
                                       data['Close'].pct_change().rolling(window=14).std())))
        data['BB_High'] = data['Close'].rolling(window=20).mean() + (2 * data['Close'].rolling(window=20).std())
        data['BB_Low'] = data['Close'].rolling(window=20).mean() - (2 * data['Close'].rolling(window=20).std())
        data['BB_Width'] = data['BB_High'] - data['BB_Low']
        data['ATR'] = data['High'] - data['Low']
        for lag in [1, 3, 7]:
            data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
            data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
        data['Rolling_Mean_7'] = data['Close'].rolling(window=7).mean()
        data['Rolling_Std_7'] = data['Close'].rolling(window=7).std()
        data['Daily_Return'] = data['Close'].pct_change()
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        data.fillna(method='bfill', inplace=True)

        # Feature selection
        features = [
            'Open', 'High', 'Low', 'Adj Close', 'Volume', 'SMA_7', 'SMA_30', 'EMA_7', 'EMA_30', 'RSI_14',
            'BB_High', 'BB_Low', 'BB_Width', 'ATR', 'Close_Lag_1', 'Close_Lag_3', 'Close_Lag_7',
            'Volume_Lag_1', 'Volume_Lag_3', 'Volume_Lag_7', 'Rolling_Mean_7', 'Rolling_Std_7',
            'Daily_Return', 'Log_Return'
        ]

        X = data[features]
        X_scaled = scaler.transform(X)
        X_selected = X[:, feature_selector.support_]

        # Prepare input for LSTM
        X_lstm = X_selected[-1].reshape(1, X_selected.shape[1], 1)

        # Predictions
        xgb_pred = xgb_model.predict(X_selected[-1].reshape(1, -1))[0]
        ridge_pred = ridge_model.predict(X_selected[-1].reshape(1, -1))[0]
        lstm_pred = lstm_model.predict(X_lstm)[0][0]

        # Display predictions
        st.write("### Predicted Next Day Prices:")
        st.write(f"**XGBoost Model:** ${xgb_pred:.2f}")
        st.write(f"**Ridge Model:** ${ridge_pred:.2f}")
        st.write(f"**LSTM Model:** ${lstm_pred:.2f}")
    else:
        st.write("Failed to retrieve data. Please check the ticker symbol.")
