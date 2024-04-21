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
frequency_latency = 2  # Increased frequency for more oscillations
latency = amplitude_throughput * 0.8 * np.sin(2 * np.pi * frequency_latency * x) + 1.2
latency += np.random.normal(0, 0.1, latency.shape)

# Introduce a spike in throughput and adjust latency
spike_length = 100  # Short spike
spike_position = 500  # Middle of the data
spike_amplitude = 3
spike_throughput = spike_amplitude * np.abs(np.sin(2 * np.pi * 10 * np.linspace(0, 1, spike_length)))
throughput[spike_position:spike_position + spike_length] += spike_throughput
latency[spike_position:spike_position + spike_length] += spike_amplitude * 0.8 * np.abs(np.sin(2 * np.pi * 20 * np.linspace(0, 1, spike_length)))

# Generate CPU and Memory usage based on throughput
cpu_usage = 0.3 + 0.05 * np.sin(2 * np.pi * 0.5 * x) + 0.05 * throughput / np.max(throughput)
memory_usage = 0.4 + 0.1 * np.sin(2 * np.pi * 0.6 * x) + 0.1 * throughput / np.max(throughput)

# Normalize CPU, Memory, Throughput, and Latency for the neural network and plotting
scaled_cpu = (cpu_usage - np.min(cpu_usage)) / (np.max(cpu_usage) - np.min(cpu_usage))
scaled_memory = (memory_usage - np.min(memory_usage)) / (np.max(memory_usage) - np.min(memory_usage))
scaled_throughput = (throughput - np.min(throughput)) / (np.max(throughput) - np.min(throughput))
scaled_latency = (latency - np.min(latency)) / (np.max(latency) - np.min(latency))

# Combine scaled metrics for LSTM input
data = np.column_stack((scaled_throughput, scaled_latency, scaled_cpu, scaled_memory))
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
    Input(shape=(n_steps, 4)),  # Update input shape for 4 features
    Bidirectional(LSTM(150, activation='relu', return_sequences=True)),
    LSTM(100, activation='relu', return_sequences=True),
    LSTM(50, activation='relu'),
    Dense(4)  # Update output layer for 4 outputs
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Fit model with the learning rate scheduler callback
model.fit(X, y, epochs=200, verbose=1, callbacks=[callback])
model.save('my_updated_lstm_model.keras')

# Making predictions
predictions = model.predict(X)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(x[:-n_steps], scaled_throughput[:-n_steps], label='Normalized Throughput', color='blue')
plt.plot(x[:-n_steps], scaled_latency[:-n_steps], label='Normalized Latency', color='red')
plt.plot(x[:-n_steps], scaled_cpu[:-n_steps], label='Normalized CPU Usage', color='green')
plt.plot(x[:-n_steps], scaled_memory[:-n_steps], label='Normalized Memory Usage', color='purple')
plt.plot(x[:-n_steps], predictions[:, 0], linestyle='--', label='Predicted Throughput', color='blue')
plt.plot(x[:-n_steps], predictions[:, 1], linestyle='--', label='Predicted Latency', color='red')
plt.plot(x[:-n_steps], predictions[:, 2], linestyle='--', label='Predicted CPU Usage', color='green')
plt.plot(x[:-n_steps], predictions[:, 3], linestyle='--',label='Predicted Memory Usage', color='purple')
plt.title('Simulation and Prediction of Throughput, Latency, CPU, and Memory Usage')
plt.xlabel('Time')
plt.ylabel('Normalized Metrics')
plt.legend()
plt.show()

# This revised script includes spikes in the throughput and latency at the specified position and propagates the effects to CPU and memory usage slightly. The model is then trained to predict these metrics based on the historic data patterns it learns, including the influence of spikes. This setup
