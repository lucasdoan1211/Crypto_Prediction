import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge

# Function to fetch and preprocess data
@st.cache
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

# Define main function
def main():
    st.title("Ridge Regression Stock Price Prediction")

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

    # Display raw data
    st.write(f"Data for {ticker}")
    st.dataframe(data)

    # Load pre-trained artifacts
    scaler = joblib.load("scaler.pkl")
    ridge_model = joblib.load("model_ridge.pkl")

    # Feature engineering
    data['SMA_7'] = data['Close'].rolling(window=7).mean()
    data['SMA_30'] = data['Close'].rolling(window=30).mean()
    data['RSI_14'] = 100 - (100 / (1 + (data['Close'].diff(1).clip(lower=0).rolling(window=14).mean() /
                                         data['Close'].diff(1).clip(upper=0).abs().rolling(window=14).mean())))
    data['Daily_Return'] = data['Close'].pct_change()
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Close_Lag_1'] = data['Close'].shift(1)
    data.dropna(inplace=True)

    # Feature selection
    optimal_features = ['Close_Lag_1', 'SMA_7', 'SMA_30', 'RSI_14', 'Daily_Return', 'Log_Return']
    X = data[optimal_features]

    # Scaling
    X_scaled = scaler.transform(X)

    # Predict using Ridge Model
    predictions = ridge_model.predict(X_scaled)

    # Display predictions
    data['Predicted_Close'] = predictions
    st.write("Predicted Closing Prices")
    st.line_chart(data[['Close', 'Predicted_Close']])

# Run the app
if __name__ == "__main__":
    main()
