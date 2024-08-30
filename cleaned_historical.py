#open, close, high, low volume
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Load data
df = pd.read_csv("BTC-USD(1).csv", parse_dates=['Date'])

# Handle missing values if there are any, may not be since it came form yahoo?
df.fillna(method='ffill', inplace=True)  # Forward fill method

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

