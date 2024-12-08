import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
import pickle
import yfinance as yf
from datetime import datetime, timedelta

# Load pre-trained models and scaler
@st.cache_resource
def load_artifacts():
    lstm_model = load_model("lstm_model.h5")
    xgb_model = pickle.load(open("xgb_model.pkl", "rb"))
    ridge_model = pickle.load(open("ridge_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return lstm_model, xgb_model, ridge_model, scaler

lstm_model, xgb_model, ridge_model, scaler = load_artifacts()

# Streamlit UI
st.title("Next-Day Prediction Deployment")
st.sidebar.header("Data Source")

# Input for ticker symbol with fixed 1-year range
ticker = st.sidebar.text_input("Enter Ticker Symbol", "AAPL")
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

if st.sidebar.button("Fetch Data and Predict Next Day"):
    try:
        # Fetch data from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date)
        st.write(f"### Data for {ticker} (Last 1 Year)")
        st.write(data.tail())

        # Preprocessing
        try:
            data_scaled = scaler.transform(data.drop(columns=['Adj Close'], errors='ignore'))
            X_lstm = data_scaled.reshape((data_scaled.shape[0], data_scaled.shape[1], 1))
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")
            st.stop()

        # Predictions
        st.write("### Next-Day Predictions")
        next_day_data = data_scaled[-1].reshape(1, -1)
        next_day_lstm = X_lstm[-1].reshape(1, X_lstm.shape[1], 1)

        ridge_prediction = ridge_model.predict(next_day_data)[0]
        xgb_prediction = xgb_model.predict(next_day_data)[0]
        lstm_prediction = lstm_model.predict(next_day_lstm).flatten()[0]

        # Combine results
        predictions = {
            "Ridge Prediction": ridge_prediction,
            "XGBoost Prediction": xgb_prediction,
            "LSTM Prediction": lstm_prediction
        }
        st.write(predictions)

    except Exception as e:
        st.error(f"Error fetching data: {e}")
else:
    st.write("Click the button to fetch data and predict the next day.")
