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
from sklearn.linear_model import Ridge
import joblib
import plotly.graph_objects as go
from datetime import date, timedelta

# Load models and feature selector
try:
    model_lstm = load_model("model_lstm.h5")
    model_xgb = joblib.load("model_xgb.pkl")
    model_ridge = joblib.load("model_ridge.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_selector = joblib.load("feature_selector.pkl")
except Exception as e:
    st.error(f"Error loading models or scaler: {e}")
    st.stop()

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
    data.reset_index(inplace=True)
    data.fillna(data.median(), inplace=True)
    return data


# Streamlit App
st.title("Crypto Price Prediction App")

# Sidebar
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Crypto Ticker (e.g., ONDO-USD)", "ONDO-USD")
# Set end_date to today and calculate start_date as 6 months before
end_date = date.today()
start_date = end_date - timedelta(days=6*30) # Approximately 6 months

prediction_days = st.sidebar.number_input("Prediction Days", min_value=1, max_value=30, value=7)
selected_model = st.sidebar.selectbox("Select Model", ["LSTM", "XGBoost", "Ridge"])

# Download and preprocess data
if st.sidebar.button("Predict"):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found for the given ticker and date range.")
            st.stop()

        data = data.reset_index()
        data_for_features = data.copy()
        data_for_features = create_features(data_for_features)

        features = [
            'Open', 'High', 'Low', 'Adj Close', 'Volume', 'SMA_7', 'SMA_30', 'EMA_7', 'EMA_30',
            'RSI_14', 'BB_High', 'BB_Low', 'BB_Width', 'ATR', 'Close_Lag_1', 'Close_Lag_3',
            'Close_Lag_7', 'Volume_Lag_1', 'Volume_Lag_3', 'Volume_Lag_7', 'Rolling_Mean_7',
            'Rolling_Std_7', 'Daily_Return', 'Log_Return'
        ]
        
        X = data_for_features[features]
        X_scaled = scaler.transform(X)
        X_selected = X.columns[feature_selector.support_]
        X_scaled_selected = scaler.transform(X[X_selected])
        
        # Make predictions
        future_dates = [data['Date'].iloc[-1] + timedelta(days=i) for i in range(1, prediction_days + 1)]
        future_predictions = []

        if selected_model == "LSTM":
            X_lstm = X_scaled_selected.reshape((X_scaled_selected.shape[0], X_scaled_selected.shape[1], 1))
            for _ in range(prediction_days):
              last_features = X_lstm[-1].reshape(1, X_lstm.shape[1], 1)
              prediction = model_lstm.predict(last_features)[0][0]
              future_predictions.append(prediction)
              
              new_features = np.append(last_features[0][1:], prediction).reshape(1, X_lstm.shape[1], 1)
              X_lstm = np.concatenate((X_lstm, new_features), axis=0)

        elif selected_model == "XGBoost":
            last_features_xgb = X_scaled_selected[-1]
            for _ in range(prediction_days):
                prediction = model_xgb.predict(last_features_xgb.reshape(1, -1))[0]
                future_predictions.append(prediction)

                # Update the last_features for the next prediction
                new_features_xgb = np.append(last_features_xgb[1:], prediction)
                last_features_xgb = new_features_xgb

        elif selected_model == "Ridge":
            last_features_ridge = X_scaled_selected[-1]
            for _ in range(prediction_days):
                prediction = model_ridge.predict(last_features_ridge.reshape(1, -1))[0]
                future_predictions.append(prediction)
                # Update the last_features for the next prediction
                new_features_ridge = np.append(last_features_ridge[1:], prediction)
                last_features_ridge = new_features_ridge

        # Create a DataFrame for predictions
        future_predictions_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': future_predictions
        })

        # Display predictions
        st.subheader(f"Predictions for {ticker} using {selected_model}")
        st.dataframe(future_predictions_df)

        # Plotting
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Historical Close Price'))
        fig.add_trace(go.Scatter(x=future_predictions_df['Date'], y=future_predictions_df['Predicted_Close'], mode='lines+markers', name='Predicted Close Price'))
        fig.update_layout(title=f'{ticker} Price Prediction', xaxis_title='Date', yaxis_title='Price (USD)')
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
