import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# Generate data
x = np.linspace(0, 10, 1000)
amplitude_throughput = 1
frequency_throughput = 1
throughput = amplitude_throughput * np.sin(2 * np.pi * frequency_throughput * x) + 1.5  # Shifted to avoid negative values
throughput += np.random.normal(0, 0.2, throughput.shape)

frequency_latency = 2  # Higher frequency for latency
latency = amplitude_throughput * 0.8 * np.sin(2 * np.pi * frequency_latency * x) + 1.2
latency += np.random.normal(0, 0.1, latency.shape)

# Introduce a spike in throughput and latency
spike_length = 100
spike_position = 500
spike_amplitude = 3
spike_throughput = spike_amplitude * np.abs(np.sin(2 * np.pi * 10 * np.linspace(0, 1, spike_length)))
throughput[spike_position:spike_position + spike_length] += spike_throughput
latency_spike = spike_amplitude * 0.8 * np.abs(np.sin(2 * np.pi * 20 * np.linspace(0, 1, spike_length)))
latency[spike_position:spike_position + spike_length] += latency_spike

# Normalize data
scaled_throughput = (throughput - np.min(throughput)) / (np.max(throughput) - np.min(throughput))
scaled_latency = (latency - np.min(latency)) / (np.max(latency) - np.min(latency))
data = np.column_stack((scaled_throughput, scaled_latency))

# Create dataset for LSTM
def create_dataset(data, n_steps=10):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

n_steps = 10
X, y = create_dataset(data, n_steps)

# Define LSTM model
model = Sequential([
    Input(shape=(n_steps, 2)),
    LSTM(50, activation='relu'),
    Dense(2)
])
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=50, verbose=1)

# Predict on the entire dataset
predictions = []
current_batch = X[:1]  # start with the first sequence for prediction
for i in range(len(data) - n_steps):
    current_pred = model.predict(current_batch)[0]
    predictions.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

# Visualization
plt.figure(figsize=(15, 5))
plt.plot(data[n_steps:, 0], label='True Normalized Throughput', color='blue')
plt.plot([x[0] for x in predictions], '--', label='Predicted Normalized Throughput', color='blue')
plt.plot(data[n_steps:, 1], label='True Normalized Latency', color='red')
plt.plot([x[1] for x in predictions], '--', label='Predicted Normalized Latency', color='red')
plt.title('LSTM Predictions vs Actual Data')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Metrics')
plt.legend()
plt.show()

