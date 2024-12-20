import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
import joblib
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
import os

# Streamlit App Title
st.title("Crypto Price Prediction with Feature Selection")
st.write(
    "This app predicts the next day's cryptocurrency closing price using Ridge, XGBoost, and LSTM models. "
    "It dynamically selects features and reuses saved models for efficient predictions."
)

# Load saved models or train new ones
try:
    scaler = joblib.load("scaler.pkl")
    feature_selector = joblib.load("feature_selector.pkl")
    ridge_model = joblib.load("model_ridge.pkl")
    xgb_model = joblib.load("model_xgb.pkl")
    lstm_model = load_model("model_lstm.h5")
    st.success("Models and scalers loaded successfully!")
except Exception as e:
    st.warning("Saved models not found. They will be trained and saved during the first run.")

# Input Section
ticker = st.text_input("Enter the Cryptocurrency Ticker (e.g., BTC-USD):", value="BTC-USD")

if st.button("Predict"):
    try:
        # Fetch Data from Yahoo Finance
        today = pd.Timestamp.today()
        start_date = today - pd.DateOffset(years=1)
        data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))

        if data.empty:
            st.error("No data found for the given ticker. Please try another ticker.")
            st.stop()

        st.write("Data loaded successfully!")

        # Feature Engineering
        data = data.reset_index()
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
        data['ATR'] = ta.volatility.average_true_range(
            high=data['High'], low=data['Low'], close=data['Close'], window=14)
        for lag in [1, 3, 7]:
            data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
            data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
        data['Rolling_Mean_7'] = data['Close'].rolling(window=7).mean()
        data['Rolling_Std_7'] = data['Close'].rolling(window=7).std()
        data['Daily_Return'] = data['Close'].pct_change()
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        data.reset_index(inplace=True)
        data.fillna(data.median(), inplace=True)

        # Feature selection with RFE
        features = [
            'Open', 'High', 'Low', 'Adj Close', 'Volume', 'SMA_7', 'SMA_30', 'EMA_7', 'EMA_30', 'RSI_14',
            'BB_High', 'BB_Low', 'BB_Width', 'ATR', 'Close_Lag_1', 'Close_Lag_3', 'Close_Lag_7',
            'Volume_Lag_1', 'Volume_Lag_3', 'Volume_Lag_7', 'Rolling_Mean_7', 'Rolling_Std_7',
            'Daily_Return', 'Log_Return'
        ]
        X = data[features]
        y = data['Close']

        if not os.path.exists("feature_selector.pkl"):
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            estimator = XGBRegressor(
                n_estimators=500, max_depth=None, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1)
            feature_selector = RFE(estimator=estimator, step=5)
            feature_selector.fit(X_scaled, y)

            # Save scaler and feature selector
            joblib.dump(scaler, "scaler.pkl")
            joblib.dump(feature_selector, "feature_selector.pkl")

        # Reload the scaler and feature selector if saved
        scaler = joblib.load("scaler.pkl")
        feature_selector = joblib.load("feature_selector.pkl")
        optimal_features = X.columns[feature_selector.support_]
        X = data[optimal_features]
        X_scaled = scaler.fit_transform(X)

        # Train and save models if not already saved
        if not os.path.exists("model_lstm.h5"):
            # LSTM Model
            X_lstm = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
            lstm_model = Sequential([
                LSTM(64, activation='relu', input_shape=(X_lstm.shape[1], 1)),
                Dense(1)
            ])
            lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            lstm_model.fit(X_lstm, y, epochs=10, batch_size=32, verbose=1)
            lstm_model.save("model_lstm.h5")

        if not os.path.exists("model_xgb.pkl"):
            # XGBoost Model
            xgb_model = XGBRegressor(
                n_estimators=500, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                random_state=42
            )
            xgb_model.fit(X_scaled, y)
            joblib.dump(xgb_model, "model_xgb.pkl")

        if not os.path.exists("model_ridge.pkl"):
            # Ridge Model
            ridge_params = {'alpha': [0.1, 1.0, 10.0]}
            ridge_model = GridSearchCV(Ridge(), param_grid=ridge_params, cv=3)
            ridge_model.fit(X_scaled, y)
            joblib.dump(ridge_model, "model_ridge.pkl")

        # Load saved models for predictions
        lstm_model = load_model("model_lstm.h5")
        xgb_model = joblib.load("model_xgb.pkl")
        ridge_model = joblib.load("model_ridge.pkl")

        # Prepare for Prediction
        latest_data = X_scaled[-1].reshape(1, -1)
        latest_data_lstm = latest_data.reshape((latest_data.shape[0], latest_data.shape[1], 1))

        # Predictions
        ridge_prediction = ridge_model.predict(latest_data)[0]
        xgb_prediction = xgb_model.predict(latest_data)[0]
        lstm_prediction = lstm_model.predict(latest_data_lstm).flatten()[0]

        # Display Predictions
        st.subheader("Predictions for the Next Day")
        st.write(f"**Ridge Regression Prediction:** {ridge_prediction:.2f}")
        st.write(f"**XGBoost Prediction:** {xgb_prediction:.2f}")
        st.write(f"**LSTM Prediction:** {lstm_prediction:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
