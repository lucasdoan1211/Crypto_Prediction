import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model
from sklearn.linear_model import Ridge
import ta
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
import joblib

st.title("Dynamic Crypto Price Prediction (10-Month Time Frame)")

# Input Section
ticker = st.text_input("Enter the Cryptocurrency Ticker (e.g., BTC-USD):", value="BTC-USD")

if st.button("Predict"):
    try:
        # Fetch Data for 10 months
        today = pd.Timestamp.today()
        start_date = today - pd.DateOffset(months=10)
        data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))

        if data.empty:
            st.error("No data found for the given ticker. Please try another ticker.")
            st.stop()

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
        data['ATR'] = ta.volatility.average_true_range(
            high=data['High'], low=data['Low'], close=data['Close'], window=14)
        for lag in [1, 3, 7]:
            data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
            data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
        data['Rolling_Mean_7'] = data['Close'].rolling(window=7).mean()
        data['Rolling_Std_7'] = data['Close'].rolling(window=7).std()
        data['Daily_Return'] = data['Close'].pct_change()
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))

        # Fill missing data
        data.fillna(method="bfill", inplace=True)

        # Features
        features = [
            'Open', 'High', 'Low', 'Adj Close', 'Volume', 'SMA_7', 'SMA_30', 'EMA_7', 'EMA_30', 'RSI_14',
            'BB_High', 'BB_Low', 'BB_Width', 'ATR', 'Close_Lag_1', 'Close_Lag_3', 'Close_Lag_7',
            'Volume_Lag_1', 'Volume_Lag_3', 'Volume_Lag_7', 'Rolling_Mean_7', 'Rolling_Std_7',
            'Daily_Return', 'Log_Return'
        ]
        X = data[features]

        # Load scaler and feature selector
        scaler_X = joblib.load("scaler_X.pkl")  # Corrected scaler loading
        scaler_y = joblib.load("scaler_y.pkl")
        selector = joblib.load("feature_selector.pkl")
        
        # Scale and select features
        X_scaled = scaler_X.transform(X)
        optimal_features = X.columns[selector.support_]
        X_selected = X[optimal_features]
        X_scaled_selected = scaler_X.transform(X_selected)

        # Validate shape
        print("Shape of X_scaled_selected:", X_scaled_selected.shape)

        # Load models
        ridge_model = joblib.load("model_ridge.pkl")
        xgb_model = joblib.load("model_xgb.pkl")
        lstm_model = load_model("model_lstm.h5")

        # Prepare for Prediction - Corrected data selection and reshaping
        # Ridge/XGBoost - Select the last row, ensuring it's a 2D array for prediction
        latest_data = X_scaled_selected[-1:, :]  # Note the -1: to keep it as 2D
        print("Latest Data Shape (Ridge/XGBoost):", latest_data.shape)

        # LSTM - Reshape to (1, num_features, 1) - sequence length is 1 now
        latest_data_lstm = latest_data.reshape((1, latest_data.shape[1], 1))
        print("LSTM Data Shape:", latest_data_lstm.shape)

        # Predictions
        ridge_prediction = scaler_y.inverse_transform(ridge_model.predict(latest_data))[0][0]
        xgb_prediction = scaler_y.inverse_transform(xgb_model.predict(latest_data).reshape(-1, 1))[0][0]
        lstm_prediction = scaler_y.inverse_transform(lstm_model.predict(latest_data_lstm))[0][0]

        # Display Predictions
        st.subheader("Predictions for the Next Day")
        st.write(f"**Ridge Regression Prediction:** {ridge_prediction:.2f}")
        st.write(f"**XGBoost Prediction:** {xgb_prediction:.2f}")
        st.write(f"**LSTM Prediction:** {lstm_prediction:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
