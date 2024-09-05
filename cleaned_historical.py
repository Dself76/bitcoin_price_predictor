#open, close, high, low volume, rsi, 


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

class RSI:
    def calculate_rsi_for_series(self, series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

class BitcoinPricePredictor:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, parse_dates=['Date'])
        self.scaler = MinMaxScaler()
        self.rsi_calculator = RSI()

    def prepare_data(self):
        print(f"Initial shape: {self.df.shape}")
        self.df.ffill(inplace=True)
        print(f"Shape after ffill: {self.df.shape}")
        
        self.df['RSI'] = self.rsi_calculator.calculate_rsi_for_series(self.df['Close'])
        print(f"Columns: {self.df.columns.tolist()}")
        
        self.df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']] = self.scaler.fit_transform(
            self.df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']]
        )
        
        print(f"Shape after scaling: {self.df.shape}")
        
        self.df['Target'] = self.df['Close'].shift(-1)
        self.df.dropna(inplace=True)
        print(f"Final shape: {self.df.shape}")
        
        if self.df.empty:
            raise ValueError("DataFrame is empty after preprocessing.")
        
        X = self.df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']]
        y = self.df['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.train_data = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
        self.test_data = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))
        
        self.train_data = self.train_data.shuffle(len(X_train)).batch(32)
        self.test_data = self.test_data.batch(32)

    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def train_model(self, epochs=50):
        self.model.fit(self.train_data, epochs=epochs, validation_data=self.test_data)

    def evaluate_model(self):
        return self.model.evaluate(self.test_data)

    def predict(self, X):
        return self.model.predict(X)

    def denormalize_prediction(self, normalized_prediction):
        dummy_array = np.zeros((1, 6))
        dummy_array[0, 3] = normalized_prediction[0]  # Assuming the prediction is a scalar or 1D array
        return self.scaler.inverse_transform(dummy_array)[0, 3]
    
   
if __name__ == "__main__":
    predictor = BitcoinPricePredictor("BTC-USD(1).csv")
    predictor.prepare_data()
    predictor.build_model()
    predictor.train_model(epochs=50)
    test_loss = predictor.evaluate_model()
    print(f"Final Test Loss: {test_loss}")

    # Example of making predictions and printing them
    X_test = predictor.df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']].iloc[-len(predictor.test_data):].values
    predictions = predictor.predict(X_test)

    denormalized_predictions = [predictor.denormalize_prediction(p) for p in predictions]

    print("Predictions:")
    for prediction in denormalized_predictions:
        print(prediction)

'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

class BitcoinPricePredictor:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, parse_dates=['Date'])
        self.scaler = MinMaxScaler()

    def calculate_rsi_for_series(self, series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def prepare_data(self):
        print(f"Initial shape: {self.df.shape}")
        
        self.df.ffill(inplace=True)
        print(f"Shape after ffill: {self.df.shape}")
        
        # Calculate RSI directly here
        self.df['RSI'] = self.calculate_rsi_for_series(self.df['Close'])
        print(f"Columns: {self.df.columns.tolist()}")
        
        self.df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']] = self.scaler.fit_transform(
            self.df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']]
        )
        print(f"Shape after scaling: {self.df.shape}")
        
        self.df['Target'] = self.df['Close'].shift(-1)
        self.df.dropna(inplace=True)
        print(f"Final shape: {self.df.shape}")
        
        if self.df.empty:
            raise ValueError("DataFrame is empty after preprocessing.")
        
        X = self.df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']]
        y = self.df['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.train_data = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
        self.test_data = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))
        
        self.train_data = self.train_data.shuffle(len(X_train)).batch(32)
        self.test_data = self.test_data.batch(32)

    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def train_model(self, epochs=50):
        self.model.fit(self.train_data, epochs=epochs, validation_data=self.test_data)

    def evaluate_model(self):
        return self.model.evaluate(self.test_data)

    def predict(self, X):
        return self.model.predict(X)

    def normalize_features(self, features):
        return self.scaler.transform(features)

    def denormalize_prediction(self, normalized_prediction):
        dummy_array = np.zeros((1, 6))
        dummy_array[0, 3] = normalized_prediction[0][0]
        return self.scaler.inverse_transform(dummy_array)[0, 3]

if __name__ == "__main__":
    predictor = BitcoinPricePredictor("BTC-USD(1).csv")
    predictor.prepare_data()
    predictor.build_model()
    predictor.train_model(epochs=50)
    test_loss = predictor.evaluate_model()
    print(f"Final Test Loss: {test_loss}")


'''

'''
# this is the function before i switched to class

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Load data
df = pd.read_csv("BTC-USD(1).csv", parse_dates=['Date'])

# Handle missing values using the forward fill method
df.ffill(inplace=True)  # Forward fill method

# Normalize the columns to help with the training
scaler = MinMaxScaler()
df[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])

# Define features and target variable, like for example, predicting the next 'Close' price
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)  # Removing the last row which now has NaN

# Split the dataset into train and test sets
X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to TensorFlow tensors
train_data = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
test_data = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))

# Batch and shuffle the data
train_data = train_data.shuffle(len(X_train)).batch(32)
test_data = test_data.batch(32)
'''