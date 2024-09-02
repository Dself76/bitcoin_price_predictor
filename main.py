import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

class RSI:
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = self.fetch_data()

    def fetch_data(self):
        stock = yf.Ticker(self.ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*10)  # Fetch data for the past 10 years
        return stock.history(start=start_date, end=end_date)

    def calculate_rsi(self, window=14):
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

    def get_rsi_series(self):
        return self.df[['Close', 'RSI']]

class BitcoinPricePredictor:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, parse_dates=['Date'])
        self.scaler = MinMaxScaler()
        # Ensure RSI is calculated if not present
        if 'RSI' not in self.df.columns:
            self.calculate_rsi()

    def calculate_rsi(self, window=14):
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

    def prepare_data(self):
        self.df.ffill(inplace=True)
        self.df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']] = self.scaler.fit_transform(
            self.df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']])
        self.df['Target'] = self.df['Close'].shift(-1)
        self.df.dropna(inplace=True)


'''
from cleaned_historical import BitcoinPricePredictor
from rsi import RSI
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def get_latest_bitcoin_data():
    btc = yf.Ticker("BTC-USD")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=75)  # Get 75 days of data to calculate RSI and have enough for prediction
    data = btc.history(start=start_date, end=end_date)
    return data

if __name__ == "__main__":
    # Load historical data and calculate RSI
    historical_data = pd.read_csv("BTC-USD(1).csv", parse_dates=['Date'])
    rsi_calculator = RSI("BTC-USD")
    rsi_calculator.calculate_rsi()
    rsi_series = rsi_calculator.get_rsi_series()

    # Add RSI to historical data
    historical_data['RSI'] = rsi_series['RSI']
    historical_data.to_csv("BTC-USD_with_RSI.csv", index=False)

    # Now create the predictor with the updated CSV file
    predictor = BitcoinPricePredictor("BTC-USD_with_RSI.csv")
    predictor.prepare_data()
    predictor.build_model()
    predictor.train_model(epochs=50)
    test_loss = predictor.evaluate_model()
    print(f"Final Test Loss: {test_loss}")
    
    latest_data = get_latest_bitcoin_data()
    
    # Calculate RSI for latest data
    latest_data['RSI'] = rsi_calculator.calculate_rsi_for_series(latest_data['Close'])
    
    latest_features = latest_data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']].iloc[-60:].values
    normalized_features = predictor.normalize_features(latest_features)
    
    input_data = np.expand_dims(normalized_features, axis=0)
    normalized_prediction = predictor.predict(input_data)
    denormalized_prediction = predictor.denormalize_prediction(normalized_prediction)

    latest_timestamp = latest_data.index[-1]
    latest_rsi = latest_data['RSI'].iloc[-1]
    print(f"Latest data timestamp: {latest_timestamp}")
    print(f"Latest RSI: {latest_rsi}")
    print(f"Current Price (at {latest_timestamp}): {latest_data['Close'].iloc[-1]}")
    print(f"Predicted Price (for next immediate price point after {latest_timestamp}): {denormalized_prediction}")

'''

'''

from cleaned_historical import BitcoinPricePredictor
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_latest_bitcoin_data():
    btc = yf.Ticker("BTC-USD")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=75)  # Get 75 days of data to calculate RSI and have enough for prediction
    data = btc.history(start=start_date, end=end_date)
    return data

if __name__ == "__main__":
    predictor = BitcoinPricePredictor("BTC-USD(1).csv")
    predictor.prepare_data()
    predictor.build_model()
    predictor.train_model(epochs=50)
    test_loss = predictor.evaluate_model()
    print(f"Final Test Loss: {test_loss}")
    
    latest_data = get_latest_bitcoin_data()
    latest_data['RSI'] = calculate_rsi(latest_data)
    
    latest_features = latest_data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']].iloc[-60:].values
    normalized_features = predictor.normalize_features(latest_features)
    
    input_data = np.expand_dims(normalized_features, axis=0)
    normalized_prediction = predictor.predict(input_data)
    denormalized_prediction = predictor.denormalize_prediction(normalized_prediction)

    latest_timestamp = latest_data.index[-1]
    latest_rsi = latest_data['RSI'].iloc[-1]
    print(f"Latest data timestamp: {latest_timestamp}")
    print(f"Latest RSI: {latest_rsi}")
    print(f"Current Price (at {latest_timestamp}): {latest_data['Close'].iloc[-1]}")
    print(f"Predicted Price (for next immediate price point after {latest_timestamp}): {denormalized_prediction}")
'''