import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Data generation based on your script
x = np.linspace(0, 10, 1000)
throughput = 1 * np.sin(2 * np.pi * 1 * x) + 1.5
throughput += np.random.normal(0, 0.2, throughput.shape)

latency = 1 * 0.8 * np.sin(2 * np.pi * 2 * x) + 1.2
latency += np.random.normal(0, 0.1, latency.shape)

# Spike
spike_length = 100
spike_position = 500
spike_amplitude = 3
spike_throughput = spike_amplitude * np.abs(np.sin(2 * np.pi * 10 * np.linspace(0, 1, spike_length)))
throughput[spike_position:spike_position + spike_length] += spike_throughput

latency_spike = spike_amplitude * 0.8 * np.abs(np.sin(2 * np.pi * 20 * np.linspace(0, 1, spike_length)))
latency[spike_position:spike_position + spike_length] += latency_spike

# Normalization
scaled_throughput = (throughput - np.min(throughput)) / (np.max(throughput) - np.min(throughput))
scaled_latency = (latency - np.min(latency)) / (np.max(latency) - np.min(latency))
data = np.column_stack((scaled_throughput, scaled_latency))

# Prepare data for LSTM
def create_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        a = data[i:(i + time_steps)]
        X.append(a)
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 10
X, y = create_dataset(data, time_steps)
X = np.reshape(X, (X.shape[0], X.shape[1], 2))

# Split data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X.shape[1], 2)),
    LSTM(100),
    Dense(2)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Prediction
y_pred = model.predict(X_test)

# Plot
plt.figure(figsize=(15, 6))
plt.plot(np.arange(len(y_train), len(y)), y_test[:, 0], label='True Normalized Throughput', color='blue')
plt.plot(np.arange(len(y_train), len(y)), y_pred[:, 0], '--', label='Predicted Normalized Throughput', color='blue')
plt.plot(np.arange(len(y_train), len(y)), y_test[:, 1], label='True Normalized Latency', color='red')
plt.plot(np.arange(len(y_train), len(y)), y_pred[:, 1], '--', label='Predicted Normalized Latency', color='red')
plt.title('Prediction of Throughput and Latency')
plt.xlabel('Time Step')
plt.ylabel('Normalized Value')
plt.legend()
plt.show()

