import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
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
st.title("Time-Series Prediction Deployment")
st.sidebar.header("Data Source")

# Input for ticker symbol with fixed 1-year range
ticker = st.sidebar.text_input("Enter Ticker Symbol", "AAPL")
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

if st.sidebar.button("Fetch Data"):
    try:
        # Fetch data from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date)
        st.write(f"### Data for {ticker} (Last 1 Year)")
        st.write(data.head())

        # Preprocessing
        try:
            data_scaled = scaler.transform(data.drop(columns=['Adj Close'], errors='ignore'))
            X_lstm = data_scaled.reshape((data_scaled.shape[0], data_scaled.shape[1], 1))
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")
            st.stop()

        # Predictions
        st.write("### Predictions")
        ridge_predictions = ridge_model.predict(data_scaled)
        xgb_predictions = xgb_model.predict(data_scaled)
        lstm_predictions = lstm_model.predict(X_lstm).flatten()

        # Combine results
        predictions = pd.DataFrame({
            "Ridge Predictions": ridge_predictions,
            "XGBoost Predictions": xgb_predictions,
            "LSTM Predictions": lstm_predictions
        }, index=data.index)
        st.write(predictions)

        # Visualization
        st.write("### Prediction Visualizations")

        # Ridge
        fig_ridge, ax_ridge = plt.subplots()
        ax_ridge.plot(predictions.index, ridge_predictions, label="Ridge", marker='o')
        ax_ridge.set_title("Ridge Predictions")
        ax_ridge.legend()
        st.pyplot(fig_ridge)

        # XGBoost
        fig_xgb, ax_xgb = plt.subplots()
        ax_xgb.plot(predictions.index, xgb_predictions, label="XGBoost", marker='o')
        ax_xgb.set_title("XGBoost Predictions")
        ax_xgb.legend()
        st.pyplot(fig_xgb)

        # LSTM
        fig_lstm, ax_lstm = plt.subplots()
        ax_lstm.plot(predictions.index, lstm_predictions, label="LSTM", marker='o')
        ax_lstm.set_title("LSTM Predictions")
        ax_lstm.legend()
        st.pyplot(fig_lstm)

    except Exception as e:
        st.error(f"Error fetching data: {e}")
else:
    st.write("Click the button to fetch data for the last
