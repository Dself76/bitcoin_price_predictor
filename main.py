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