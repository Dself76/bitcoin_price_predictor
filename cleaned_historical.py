#open, close, high, low volume

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

class BitcoinPricePredictor:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, parse_dates=['Date'])
        self.scaler = MinMaxScaler()

    def prepare_data(self):
        print(f"Initial shape: {self.df.shape}")
        
        self.df.ffill(inplace=True)
        print(f"Shape after ffill: {self.df.shape}")
        
        if 'RSI' not in self.df.columns:
            self.df['RSI'] = 0
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

    # Include other methods (build_model, train_model, evaluate_model, predict, etc.) here


'''
#below only adds to prediction with out using rsi
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

class BitcoinPricePredictor:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, parse_dates=['Date'])
        self.scaler = MinMaxScaler()

    def prepare_data(self):
        # Handle missing values
        self.df.ffill(inplace=True)  # Forward fill method
        
        # Normalize the columns
        self.df[['Open', 'High', 'Low', 'Close', 'Volume']] = self.scaler.fit_transform(
            self.df[['Open', 'High', 'Low', 'Close', 'Volume']]
        )

        # Create target variable
        self.df['Target'] = self.df['Close'].shift(-1)
        self.df.dropna(inplace=True)  # Remove the last row which now has NaN

        # Define features and target variable
        X = self.df[['Open', 'High', 'Low', 'Close', 'Volume']]
        y = self.df['Target']

        # Split the dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert to TensorFlow tensors
        self.train_data = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
        self.test_data = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))

        # Batch and shuffle the data
        self.train_data = self.train_data.shuffle(len(X_train)).batch(32)
        self.test_data = self.test_data.batch(32)

    def build_model(self):
        # Define the neural network model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),  # 5 input features
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)  # Output layer for regression
        ])

        # Compile the model
        self.model.compile(optimizer='adam', loss='mse')

    def train_model(self, epochs=50):
        # Train the model
        self.history = self.model.fit(self.train_data, epochs=epochs, validation_data=self.test_data)

    def evaluate_model(self):
        # Evaluate the model
        loss = self.model.evaluate(self.test_data)
        print(f'Test Loss: {loss}')
        return loss

    def predict(self, X):
        if isinstance(X, np.ndarray) and X.shape == (1, 5):
            return self.model.predict(X)
        else:
            raise ValueError("Input should be a numpy array with shape (1, 5)")

    def normalize_features(self, features):
        return self.scaler.transform(features)

    def denormalize_prediction(self, normalized_prediction):
        # Create a dummy array with the same shape as the original feature set
        dummy_array = np.zeros((1, 5))
        
        # Place the normalized prediction in the correct position (assuming 'Close' is the 4th column)
        dummy_array[0, 3] = normalized_prediction[0][0]
        
        # Inverse transform the entire dummy array
        denormalized_array = self.scaler.inverse_transform(dummy_array)
        
        # Return only the 'Close' price
        return denormalized_array[0, 3]

# Example usage
if __name__ == "__main__":
    predictor = BitcoinPricePredictor("BTC-USD(1).csv")
    predictor.prepare_data()
    predictor.build_model()
    predictor.train_model(epochs=50)
    predictor.evaluate_model()

    # Example of making a prediction
    #  get the latest features
    latest_features = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]])  # Dummy normalized data
    normalized_prediction = predictor.predict(latest_features)
    denormalized_prediction = predictor.denormalize_prediction(normalized_prediction)
    print(f"Predicted Price: {denormalized_prediction}")


    
# Example usage of how i can put it, but use it main and add other data...
if __name__ == "__main__":
    predictor = BitcoinPricePredictor("BTC-USD(1).csv")
    predictor.prepare_data()
    predictor.build_model()
    predictor.train_model(epochs=50)
    predictor.evaluate_model()
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