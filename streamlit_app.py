import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
import ta
import joblib
from sklearn.preprocessing import RobustScaler

# Load the trained model and scaler
ridge_model = joblib.load("model_ridge.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("feature_selector.pkl")

# Function to preprocess data
def preprocess_data(data):
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
    data['ATR'] = ta.volatility.average_true_range(high=data['High'], low=data['Low'], close=data['Close'], window=14)
    for lag in [1, 3, 7]:
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
        data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
    data['Rolling_Mean_7'] = data['Close'].rolling(window=7).mean()
    data['Rolling_Std_7'] = data['Close'].rolling(window=7).std()
    data['Daily_Return'] = data['Close'].pct_change()
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)

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

# Function to predict the next day's price
def predict_next_day_price(ticker):
    # Download 6 months of data
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(months=6)
    data = yf.download(ticker, start=start_date, end=end_date)

    # Preprocess the data
    X_latest = preprocess_data(data)

    # Predict using the Ridge model
    prediction = ridge_model.predict(X_latest[-1].reshape(1, -1))

    return prediction[0]

# Streamlit app
st.title("Cryptocurrency Price Prediction (Ridge Model)")

# Sidebar
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., BTC-USD)", "BTC-USD")

# Prediction button
if st.sidebar.button("Predict Next Day Price"):
    if ticker:
        try:
            with st.spinner("Predicting..."):
                predicted_price = predict_next_day_price(ticker)
            st.sidebar.success(f"Predicted Price for {ticker}: ${predicted_price:,.2f}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    else:
        st.sidebar.warning("Please enter a ticker symbol.")
