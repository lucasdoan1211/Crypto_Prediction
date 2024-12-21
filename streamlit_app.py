import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
import joblib
from xgboost import XGBRegressor
import ta  # Import for technical analysis

# Load saved models and preprocessing objects
scaler = joblib.load("scaler.pkl")
feature_selector = joblib.load("feature_selector.pkl")
lstm_model = load_model("model_lstm.h5")
xgb_model = joblib.load("model_xgb.pkl")

# Streamlit app
def main():
    st.title("Stock Price Prediction App")

    st.markdown("Enter the stock ticker to predict the next day's closing price based on the past 6 months of data.")

    ticker = st.text_input("Stock Ticker", "AAPL")

    if st.button("Predict"):
        try:
            # Fetch 6 months of data from Yahoo Finance
            data = yf.download(ticker, period="6mo")
            if data.empty:
                st.error("No data found for the entered ticker. Please check the ticker symbol.")
                return

            # Feature engineering
            data['SMA_7'] = data['Close'].rolling(window=7).mean()
            data['SMA_30'] = data['Close'].rolling(window=30).mean()
            data['EMA_7'] = data['Close'].ewm(span=7, adjust=False).mean()
            data['EMA_30'] = data['Close'].ewm(span=30, adjust=False).mean()
            data['RSI_14'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
            bb_indicator = ta.volatility.BollingerBands(close=data['Close'], window=20, window_dev=2)
            data['BB_High'] = bb_indicator.bollinger_hband()
            data['BB_Low'] = bb_indicator.bollinger_lband()
            data['BB_Width'] = data['BB_High'] - data['BB_Low']
            data['ATR'] = ta.volatility.average_true_range(
                high=data['High'], low=data['Low'], close=data['Close'], window=14
            )
            for lag in [1, 3, 7]:
                data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
                data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
            data['Rolling_Mean_7'] = data['Close'].rolling(window=7).mean()
            data['Rolling_Std_7'] = data['Close'].rolling(window=7).std()
            data['Daily_Return'] = data['Close'].pct_change()
            data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))

            # Handle missing values
            data.fillna(data.median(), inplace=True)

            # Features for prediction
            features = [
                'Open', 'High', 'Low', 'Adj Close', 'Volume', 'SMA_7', 'SMA_30', 'EMA_7', 'EMA_30', 'RSI_14',
                'BB_High', 'BB_Low', 'BB_Width', 'ATR', 'Close_Lag_1', 'Close_Lag_3', 'Close_Lag_7',
                'Volume_Lag_1', 'Volume_Lag_3', 'Volume_Lag_7', 'Rolling_Mean_7', 'Rolling_Std_7',
                'Daily_Return', 'Log_Return'
            ]

            X = data[features]
            X_scaled = scaler.transform(X)

            # Feature selection
            X_selected = X_scaled[:, feature_selector.support_]

            # Reshape for LSTM
            X_lstm = X_selected.reshape((X_selected.shape[0], X_selected.shape[1], 1))

            # Predict using LSTM
            lstm_pred = lstm_model.predict(X_lstm[-1].reshape(1, X_lstm.shape[1], 1))

            # Predict using XGBoost
            xgb_pred = xgb_model.predict(X_selected[-1].reshape(1, -1))

            st.write("### Prediction")
            st.write(f"Predicted Next Close Price: ${xgb_pred[0]:.2f}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
