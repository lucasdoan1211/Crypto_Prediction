import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
from tensorflow.keras.models import load_model
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator

# Load Pre-trained Models
ridge_model = joblib.load("model_ridge.pkl")
xgb_model = joblib.load("model_xgb.pkl")
lstm_model = load_model("model_lstm.h5")

# Streamlit App Title
st.title("Crypto Price Prediction with Feature Selection")
st.write("This app dynamically selects features using Recursive Feature Elimination (RFE) and predicts the next day's cryptocurrency closing price using Ridge, XGBoost, and LSTM models.")

# Input Section
ticker = st.text_input("Enter the Cryptocurrency Ticker (e.g., ONDO-USD):", value="BTC-USD")

if st.button("Predict"):
    try:
        # Fetch Data from Yahoo Finance
        today = pd.Timestamp.today()
        start_date = today - pd.DateOffset(years=1)
        data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
        
        if data.empty:
            st.error("No data found for the given ticker. Please try another ticker.")
        else:
            st.write("Data loaded successfully!")

            # Feature Engineering
            data['Date'] = pd.to_datetime(data.index)
            data.set_index('Date', inplace=True)

            # Moving Averages (SMA and EMA)
            data['SMA_7'] = SMAIndicator(close=data['Close'], window=7).sma_indicator()
            data['SMA_30'] = SMAIndicator(close=data['Close'], window=30).sma_indicator()
            data['EMA_7'] = EMAIndicator(close=data['Close'], window=7).ema_indicator()
            data['EMA_30'] = EMAIndicator(close=data['Close'], window=30).ema_indicator()

            # Relative Strength Index (RSI)
            data['RSI_14'] = RSIIndicator(close=data['Close'], window=14).rsi()

            # Bollinger Bands
            bb_indicator = BollingerBands(close=data['Close'], window=20, window_dev=2)
            data['BB_High'] = bb_indicator.bollinger_hband()
            data['BB_Low'] = bb_indicator.bollinger_lband()
            data['BB_Width'] = data['BB_High'] - data['BB_Low']

            # Average True Range (ATR)
            data['ATR'] = (data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min())

            # Lag Features
            lags = [1, 3, 7]
            for lag in lags:
                data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
                data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)

            # Rolling Statistics
            data['Rolling_Mean_7'] = data['Close'].rolling(window=7).mean()
            data['Rolling_Std_7'] = data['Close'].rolling(window=7).std()

            # Returns
            data['Daily_Return'] = data['Close'].pct_change()
            data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))

            # Drop NaN Values
            data.dropna(inplace=True)

            # Define Features and Target
            features = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'SMA_7', 'SMA_30', 'EMA_7', 'EMA_30',
                        'RSI_14', 'BB_High', 'BB_Low', 'BB_Width', 'ATR', 'Close_Lag_1', 'Close_Lag_3', 
                        'Close_Lag_7', 'Volume_Lag_1', 'Volume_Lag_3', 'Volume_Lag_7', 
                        'Rolling_Mean_7', 'Rolling_Std_7', 'Daily_Return', 'Log_Return']
            target = 'Close'

            X = data[features]
            y = data[target]

            # Split Data into Train and Test Sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale Features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Feature Selection with RFE
            estimator = XGBRegressor(
                n_estimators=500,
                max_depth=None,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            selector = RFE(estimator=estimator, step=5)
            selector = selector.fit(X_train_scaled, y_train)

            # Optimal Features
            optimal_features = [features[i] for i in range(len(features)) if selector.support_[i]]
            st.write(f"Selected features: {optimal_features}")

            # Prepare Latest Data for Prediction Using Selected Features
            latest_data = data[optimal_features].iloc[-1].values.reshape(1, -1)
            st.write("Latest Data Shape Before Scaling:", latest_data.shape)

            scaled_data = scaler.transform(latest_data)
            st.write("Scaled Data Shape:", scaled_data.shape)

            # Predictions
            ridge_prediction = ridge_model.predict(scaled_data)[0]
            xgb_prediction = xgb_model.predict(scaled_data)[0]

            # Reshape for LSTM (3D input: samples, timesteps, features)
            lstm_data = scaled_data.reshape(1, 1, scaled_data.shape[1])
            lstm_prediction = lstm_model.predict(lstm_data).flatten()[0]

            # Display Results
            st.subheader("Predictions for Next Day Closing Price:")
            st.write(f"**Ridge Regression Prediction:** ${ridge_prediction:.2f}")
            st.write(f"**XGBoost Prediction:** ${xgb_prediction:.2f}")
            st.write(f"**LSTM Prediction:** ${lstm_prediction:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
