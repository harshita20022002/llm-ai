import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 1. Load Data
ticker = 'AAPL'  # You can replace with any stock symbol
data = yf.download(ticker, start='2010-01-01', end='2024-12-31')

# 2. Preprocess
data = data[['Close']]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 3. Create Sequences
sequence_length = 60
x = []
y = []

for i in range(sequence_length, len(scaled_data)):
    x.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])

x, y = np.array(x), np.array(y)

# 4. Split into Train/Test
split = int(0.8 * len(x))
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

# 5. Build Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Train
model.fit(x_train, y_train, epochs=10, batch_size=32)

# 7. Predict
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
real = scaler.inverse_transform(y_test)

# 8. Plot
plt.figure(figsize=(10,6))
plt.plot(real, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
