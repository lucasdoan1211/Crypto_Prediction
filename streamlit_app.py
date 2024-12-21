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

# Load saved objects
scaler = joblib.load("scaler.pkl")
optimal_features = joblib.load("selected_features.pkl")
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
                high=data['High'], low=data['Low'], close=data['Close'], window=14
            )
            for lag in [1, 3, 7]:
                data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
                data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
            data['Rolling_Mean_7'] = data['Close'].rolling(window=7).mean()
            data['Rolling_Std_7'] = data['Close'].rolling(window=7).std()
            data['Daily_Return'] = data['Close'].pct_change()
            data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
            data.fillna(data.median(), inplace=True)

            # Select features
            X = data[optimal_features]
            X_scaled = scaler.transform(X)

            # Prepare input for models
            last_row_xgb = X_scaled[-1].reshape(1, -1)  # 2D for XGBoost
            last_row_lstm = X_scaled[-1].reshape(1, X_scaled.shape[1], 1)  # 3D for LSTM

            # Make predictions
            xgb_pred = xgb_model.predict(last_row_xgb)
            lstm_pred = lstm_model.predict(last_row_lstm)

            # Display result
            st.write("### Prediction")
            st.write(f"Predicted Next Close Price (XGBoost): ${xgb_pred[0]:.2f}")
            st.write(f"Predicted Next Close Price (LSTM): ${lstm_pred[0][0]:.2f}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
