
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

class RSI:
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = self.fetch_data()

    def fetch_data(self):
        stock = yf.Ticker(self.ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*10)
        hist_data = stock.history(start=start_date, end=end_date)
        return hist_data

    def calculate_rsi(self, window=14):
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

    def get_rsi_series(self):
        return self.df[['Close', 'RSI']]
    
    def calculate_rsi_for_series(self, series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


'''

#this below work can revert
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

class RSI:
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = self.fetch_data()

    def fetch_data(self):
        """Fetch historical data for the stock over the past 10 years."""
        stock = yf.Ticker(self.ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*10)  # 10 years
        hist_data = stock.history(start=start_date, end=end_date)
        return hist_data

    def calculate_rsi(self, window=14):
        """Calculate RSI for each 14-day period over the dataset."""
        rsi_values = []
        for start in range(len(self.df) - window + 1):
            end = start + window
            data = self.df['Close'].iloc[start:end]
            delta = data.diff()
            gain = delta.where(delta > 0, 0).sum() / window
            loss = -delta.where(delta < 0, 0).sum() / window
            rs = gain / loss if loss != 0 else 0
            rsi = 100 - (100 / (1 + rs)) if loss != 0 else 100
            rsi_values.append(rsi)
        
        # Pad the beginning of the RSI list with NaN to match the length of the original data
        self.df['RSI'] = [None] * (window - 1) + rsi_values

    def get_rsi_series(self):
        """Return the RSI series."""
        return self.df[['Close', 'RSI']]

def main():
    ticker_symbol = input("Enter the stock ticker symbol (e.g., AAPL for Apple): ").upper()
    try:
        rsi_calculator = RSI(ticker_symbol)
        rsi_calculator.calculate_rsi()
        rsi_series = rsi_calculator.get_rsi_series()
        print(rsi_series)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check if the ticker symbol is correct and try again.")

if __name__ == "__main__":
    main()
'''