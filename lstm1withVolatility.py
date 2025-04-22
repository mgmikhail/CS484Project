import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
filePath = r"C:/Users/darth/Downloads/HistoricalData_1741813699330.csv"
df = pd.read_csv(filePath)

# Ensure the dataset contains the required columns
expectedColumns = {'Date', 'Close/Last'}
if not expectedColumns.issubset(df.columns):
    raise ValueError(f"Missing expected columns. Found: {df.columns}")

# Reverse dataset to ensure chronological order
df = df[::-1].reset_index(drop=True)

# Convert date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Fill in missing dates with forward-filled closing prices
df = df.set_index('Date').asfreq('B', method='ffill').reset_index()

# Compute time differences in days (normalized)
df['TimeDelta'] = df['Date'].diff().dt.days.fillna(1)
df['TimeDelta'] = df['TimeDelta'] / df['TimeDelta'].max()

# Compute EMA and normalize it
df['EMA10'] = df['Close/Last'].ewm(span=10, adjust=False).mean()

#compute volatility
df['Volatility'] = df['Close/Last'].rolling(window=10).std().fillna(0)

scaler = MinMaxScaler()
df[['Close/Last', 'EMA10', 'Volatility']] = scaler.fit_transform(df[['Close/Last', 'EMA10', 'Volatility']])



# Extract closing prices, time deltas, EMA, and volatility
closingPrices = df[['Close/Last', 'TimeDelta', 'EMA10', 'Volatility']].values

# Function to create sequences of past closing prices, time gaps, and EMA
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])  # Last `seq_length` days of closing price, time gaps, and EMA
        labels.append(data[i+seq_length, 0])  # Predict only the closing price
    return np.array(sequences), np.array(labels)

# Define sequence length
seqLength = 100
X, y = create_sequences(closingPrices, seqLength)

# Split data into training and testing sets (20% test size for more evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape the data to fit LSTM input requirements
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))  # Now includes 'Close/Last', 'TimeDelta', 'EMA10', and 'Volatility'
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))

# Define the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seqLength, 4)),  # Now taking 4 features per time step
    Dropout(0.3),
    LSTM(50, return_sequences=False),
    Dropout(0.3),
    Dense(25, activation='relu'),
    Dense(1)  # Output layer predicting next day's closing price
])

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

# Train the model
model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model and scaler
model.save("lstm_stock_model_with_volatility.h5")
joblib.dump(scaler, "scaler_with_volatility.pkl")

# Load the saved model
model = load_model("lstm_stock_model_with_volatility.h5", compile=False)
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
scaler = joblib.load("scaler_with_volatility.pkl")

# Make predictions on the test set
predictions = model.predict(X_test).flatten()

y_test_reshaped = y_test.reshape(-1, 1)
predictions_reshaped = predictions.reshape(-1, 1)

# Stack with 2 dummy columns to make 3-column input
dummy_test = np.concatenate((y_test_reshaped, np.zeros((len(y_test), 2))), axis=1)
y_test_rescaled = scaler.inverse_transform(dummy_test)[:, 0]

dummy_preds = np.concatenate((predictions_reshaped, np.zeros((len(predictions), 2))), axis=1)
predictions_rescaled = scaler.inverse_transform(dummy_preds)[:, 0]



# Evaluate the model using performance metrics
mse = np.mean((predictions_rescaled - y_test_rescaled) ** 2)
mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
r2 = r2_score(y_test_rescaled, predictions_rescaled)

# Print evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Plot actual vs. predicted closing prices
plt.figure(figsize=(10, 5))
plt.plot(y_test_rescaled, label='Actual Prices', color='blue')
plt.plot(predictions_rescaled, label='Predicted Prices', color='red')
plt.legend()
plt.title('Stock Price Prediction vs Actual Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

# Function to predict the next day's closing price
def predict_next_day(last_closing_prices, last_time_deltas, last_ema_values):
    last_data = np.column_stack((last_closing_prices, last_time_deltas, last_ema_values))
    last_data_scaled = scaler.transform(last_data)
    sequence = np.array([last_data_scaled[-seqLength:]]).reshape(1, seqLength, 3)
    prediction_scaled = model.predict(sequence)
    prediction_rescaled = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
    return prediction_rescaled[0, 0]

print("Model is ready for next-day closing price prediction.")