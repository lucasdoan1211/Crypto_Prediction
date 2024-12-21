import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler

# Function to fetch and preprocess data
@st.cache
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

def feature_engineering(data):
    # Add technical indicators
    data['SMA_7'] = data['Close'].rolling(window=7).mean()
    data['SMA_30'] = data['Close'].rolling(window=30).mean()
    data['EMA_7'] = data['Close'].ewm(span=7).mean()
    data['EMA_30'] = data['Close'].ewm(span=30).mean()
    data['RSI_14'] = 100 - (100 / (1 + (data['Close'].diff(1).clip(lower=0).rolling(window=14).mean() /
                                         data['Close'].diff(1).clip(upper=0).abs().rolling(window=14).mean())))
    data['BB_High'] = data['Close'].rolling(window=20).mean() + 2 * data['Close'].rolling(window=20).std()
    data['BB_Low'] = data['Close'].rolling(window=20).mean() - 2 * data['Close'].rolling(window=20).std()
    data['BB_Width'] = data['BB_High'] - data['BB_Low']
    data['ATR'] = (data['High'] - data['Low']).rolling(window=14).mean()
    for lag in [1, 3, 7]:
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
        data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
    data['Rolling_Mean_7'] = data['Close'].rolling(window=7).mean()
    data['Rolling_Std_7'] = data['Close'].rolling(window=7).std()
    data['Daily_Return'] = data['Close'].pct_change()
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)
    return data

def main():
    st.title("Next Day Stock Price Prediction with Ridge Regression")

    # Sidebar for user input
    st.sidebar.header("Input Parameters")
    ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., BTC-USD)", "BTC-USD")
    months = st.sidebar.slider("Number of Months of Data", 1, 12, 6)

    # Fetch data
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(months=months)
    data = load_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if data.empty:
        st.error("No data found for the specified ticker and date range.")
        return

    # Feature engineering
    data = feature_engineering(data)

    # Display raw data
    st.write(f"Data for {ticker}")
    st.dataframe(data)

    # Load pre-trained artifacts
    scaler = joblib.load("scaler.pkl")
    ridge_model = joblib.load("model_ridge.pkl")

    # Features for prediction
    features = [
        'Open', 'High', 'Low', 'Adj Close', 'Volume', 'SMA_7', 'SMA_30', 'EMA_7', 'EMA_30', 'RSI_14',
        'BB_High', 'BB_Low', 'BB_Width', 'ATR', 'Close_Lag_1', 'Close_Lag_3', 'Close_Lag_7',
        'Volume_Lag_1', 'Volume_Lag_3', 'Volume_Lag_7', 'Rolling_Mean_7', 'Rolling_Std_7',
        'Daily_Return', 'Log_Return'
    ]
    X = data[features]

    # Scaling
    X_scaled = scaler.transform(X)

    # Predict next day's closing price
    latest_features = X_scaled[-1].reshape(1, -1)
    next_day_close = ridge_model.predict(latest_features)[0]

    # Display prediction
    st.write("Predicted Next Day Close Price")
    st.metric(label="Next Day Close", value=f"${next_day_close:.2f}")

    # Plot actual vs predicted
    data['Predicted_Close'] = ridge_model.predict(X_scaled)
    st.line_chart(data[['Close', 'Predicted_Close']])

if __name__ == "__main__":
    main()
