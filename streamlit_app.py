import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import numpy as np

# Function to retrieve stock data
def get_stock_data(ticker, period="6mo"):
    data = yf.download(ticker, period=period)
    return data

# Function to train a simple linear regression model
def train_model(data):
    data["Next_Close"] = data["Close"].shift(-1)
    data = data.dropna()

    X = data[["Close"]]
    y = data["Next_Close"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Fixed RMSE calculation

    return model, rmse

# Predict the next day's price
def predict_next_day(model, last_close):
    return model.predict(np.array([[last_close]]))[0]

# Streamlit app
st.title("Stock Price Prediction App")
st.write("Retrieve 6 months of stock data and predict the next day's closing price.")

# Input for the stock ticker symbol
ticker = st.text_input("Enter the stock ticker (e.g., AAPL, TSLA):", value="AAPL")

if st.button("Predict"):
    try:
        # Fetch data
        st.write("Fetching data for ticker:", ticker)
        data = get_stock_data(ticker)

        if data.empty:
            st.error("No data found for the ticker symbol. Please check and try again.")
        else:
            st.write("Data Retrieved Successfully!")
            st.write(data.tail())

            # Train model
            st.write("Training the model...")
            model, rmse = train_model(data)

            st.write(f"Model trained. RMSE: {rmse:.2f}")

            # Predict next day's price
            last_close = data["Close"].iloc[-1]
            next_day_price = predict_next_day(model, last_close)

            st.write(f"The predicted next day's closing price for {ticker} is: ${next_day_price:.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
