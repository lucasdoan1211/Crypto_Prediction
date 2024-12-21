import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
import joblib
from xgboost import XGBRegressor
import ta  
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator

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

            data.reset_index(inplace=True)

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

            # Flatten data for XGBoost
            last_row_xgb = X_selected[-1].reshape(1, -1)  # Ensure 2D for prediction

            # Reshape for LSTM
            last_row_lstm = X_selected[-1].reshape(1, X_selected.shape[1], 1)  # Ensure 3D for LSTM

            # Predict using LSTM
            lstm_pred = lstm_model.predict(last_row_lstm)

            # Predict using XGBoost
            xgb_pred = xgb_model.predict(last_row_xgb)

            st.write("### Prediction")
            st.write(f"Predicted Next Close Price: ${xgb_pred[0]:.2f}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
