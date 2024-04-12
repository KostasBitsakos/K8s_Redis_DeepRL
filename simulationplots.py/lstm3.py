import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

def create_dataset(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# Time vector
x = np.linspace(0, 10, 1000)

# Throughput simulation parameters
amplitude_throughput = 1
frequency_throughput = 1

# Generate sinusoidal throughput
throughput = amplitude_throughput * np.sin(2 * np.pi * frequency_throughput * x) + 1.5  # Shifted to avoid negative values

# Adding Gaussian noise to throughput
throughput_noise = np.random.normal(0, 0.2, throughput.shape)
throughput += throughput_noise

# Latency simulation parameters based on throughput
# Adding different pattern: more frequent oscillations and higher scaling
frequency_latency = 2  # Increased frequency for more oscillations
latency = amplitude_throughput * 0.8 * np.sin(2 * np.pi * frequency_latency * x) + 1.2  # Different base pattern
latency += np.random.normal(0, 0.1, latency.shape)  # Adding noise

# Introduce a spike in throughput and adjust latency
spike_length = 100  # Short spike
spike_position = 500  # Middle of the data
spike_amplitude = 3
spike_throughput = spike_amplitude * np.abs(np.sin(2 * np.pi * 10 * np.linspace(0, 1, spike_length)))
throughput[spike_position:spike_position + spike_length] += spike_throughput

# Increase latency during the spike with a different pattern
latency_spike = spike_amplitude * 0.8 * np.abs(np.sin(2 * np.pi * 20 * np.linspace(0, 1, spike_length)))
latency[spike_position:spike_position + spike_length] += latency_spike

# Normalize throughput and latency for the neural network and plotting
scaled_throughput = (throughput - np.min(throughput)) / (np.max(throughput) - np.min(throughput))
scaled_latency = (latency - np.min(latency)) / (np.max(latency) - np.min(latency))

# Combine scaled throughput and latency for LSTM input
data = np.column_stack((scaled_throughput, scaled_latency))
n_steps = 10
X, y = create_dataset(data, n_steps)

# Learning rate schedule function
def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    else:
        return lr * np.exp(-0.1)

callback = LearningRateScheduler(scheduler)

# Model definition
model = Sequential([
    Input(shape=(n_steps, 2)),
    Bidirectional(LSTM(150, activation='relu', return_sequences=True)),
    LSTM(100, activation='relu', return_sequences=True),
    LSTM(50, activation='relu'),
    Dense(2)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Fit model with the learning rate scheduler callback
model.fit(X, y, epochs=200, verbose=1, callbacks=[callback])

# Making predictions
predictions = model.predict(X)
model_file = 'my_lstm_model.keras'
model.save(model_file)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(x[:-n_steps], scaled_throughput[:-n_steps], label='Normalized Throughput', color='blue')
plt.plot(x[:-n_steps], scaled_latency[:-n_steps], label='Normalized Latency', color='red')
plt.plot(x[:-n_steps], predictions[:, 0], label='Predicted Throughput', color='green', linestyle='--')
plt.plot(x[:-n_steps], predictions[:, 1], label='Predicted Latency', color='purple', linestyle='--')
plt.title('Throughput and Latency Simulation and Prediction for Redis Cluster')
plt.xlabel('Time')
plt.ylabel('Normalized Metrics')
plt.legend()
plt.show()

