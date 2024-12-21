import streamlit as st
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import RobustScaler
from datetime import datetime, timedelta

FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', '52_Week_High', '52_Week_Low', 'Market_Cap', 'Beta', 'Dividend_Yield']

def fetch_stock_data(ticker):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=6*30)
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    data.reset_index(inplace=True)

    stock = yf.Ticker(ticker)
    info = stock.info

    data['52_Week_High'] = info.get('fiftyTwoWeekHigh', 0)
    data['52_Week_Low'] = info.get('fiftyTwoWeekLow', 0)
    data['Market_Cap'] = info.get('marketCap', 0)
    data['Beta'] = info.get('beta', 0)
    data['Dividend_Yield'] = info.get('dividendYield', 0)

    return data

def main():
    st.title("Stock Price Prediction App")
    
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")

    if st.button("Predict"):
        try:
            data = fetch_stock_data(ticker)
            X_next_day = data[FEATURES].iloc[-1:].values

            lstm_model = load_model("lstm_model.h5", custom_objects={"MeanSquaredError": MeanSquaredError})
            xgb_model = XGBRegressor()
            xgb_model.load_model("xgb_model.json")
            ridge_model = joblib.load("ridge_model.pkl")
            scaler = joblib.load("scaler.pkl")

            X_next_day_scaled = scaler.transform(X_next_day)
            X_next_day_lstm = X_next_day_scaled.reshape((X_next_day_scaled.shape[0], X_next_day_scaled.shape[1], 1))

            lstm_prediction = lstm_model.predict(X_next_day_lstm, verbose=0).flatten()[0]
            xgb_prediction = xgb_model.predict(X_next_day_scaled).flatten()[0]
            ridge_prediction = ridge_model.predict(X_next_day_scaled).flatten()[0]

            st.subheader("Predicted Next Day Close Price:")
            st.write(f"LSTM Model: {lstm_prediction:.2f}")
            st.write(f"XGBoost Model: {xgb_prediction:.2f}")
            st.write(f"Ridge Model: {ridge_prediction:.2f}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
