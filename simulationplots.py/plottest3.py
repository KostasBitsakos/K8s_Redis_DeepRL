import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Load the pre-trained model
model = load_model('my_updated_lstm_model.keras')
print(tf.__version__)

# Time vector for the data simulation
x = np.linspace(0, 10, 1000)

# Parameters for throughput and latency
amplitude_throughput = 1
frequency_throughput = 1
amplitude_latency = 0.8
frequency_latency = 2

# Generate sinusoidal data for throughput and latency
throughput = amplitude_throughput * np.sin(2 * np.pi * frequency_throughput * x) + 1.5
latency = amplitude_latency * np.sin(2 * np.pi * frequency_latency * x) + 1.2

# Simulate CPU and Memory usage as influenced by the throughput
cpu_usage = 0.3 + 0.05 * np.sin(2 * np.pi * 0.5 * x) + 0.05 * throughput / np.max(throughput)
memory_usage = 0.4 + 0.1 * np.sin(2 * np.pi * 0.6 * x) + 0.1 * throughput / np.max(throughput)

# Add random noise to the data
throughput += np.random.normal(0, 0.2, throughput.shape)
latency += np.random.normal(0, 0.1, latency.shape)
cpu_usage += np.random.normal(0, 0.02, cpu_usage.shape)
memory_usage += np.random.normal(0, 0.02, memory_usage.shape)

# Introduce a spike to the data at the middle point
spike_length = 100
spike_position = 500
spike_amplitude = 3
throughput[spike_position:spike_position + spike_length] += spike_amplitude * np.abs(np.sin(2 * np.pi * 10 * np.linspace(0, 1, spike_length)))
latency[spike_position:spike_position + spike_length] += spike_amplitude * 0.8 * np.abs(np.sin(2 * np.pi * 20 * np.linspace(0, 1, spike_length)))
cpu_usage[spike_position:spike_position + spike_length] += 0.05 * spike_amplitude * np.abs(np.sin(2 * np.pi * 10 * np.linspace(0, 1, spike_length)))
memory_usage[spike_position:spike_position + spike_length] += 0.05 * spike_amplitude * np.abs(np.sin(2 * np.pi * 10 * np.linspace(0, 1, spike_length)))

# Normalize the data for model prediction
scaled_throughput = (throughput - np.min(throughput)) / (np.max(throughput) - np.min(throughput))
scaled_latency = (latency - np.min(latency)) / (np.max(latency) - np.min(latency))
scaled_cpu = (cpu_usage - np.min(cpu_usage)) / (np.max(cpu_usage) - np.min(cpu_usage))
scaled_memory = (memory_usage - np.min(memory_usage)) / (np.max(memory_usage) - np.min(memory_usage))

# Combine and reshape data for the LSTM
data = np.column_stack((scaled_throughput, scaled_latency, scaled_cpu, scaled_memory))
n_steps = 10
input_seq = np.array([data[i:i+n_steps] for i in range(len(data) - n_steps)])

# Predict using the LSTM model
predictions = model.predict(input_seq)

# Plotting the results
plt.figure(figsize=(15, 7))
plt.subplot(2, 2, 1)
plt.plot(x[:-n_steps], scaled_throughput[:-n_steps], color='b', linestyle='-', label='Actual Throughput')
plt.plot(x[:-n_steps], predictions[:, 0], color='r', linestyle='--', label='Predicted Throughput')
plt.title('Throughput Prediction Comparison')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x[:-n_steps], scaled_latency[:-n_steps], color='g', linestyle='-', label='Actual Latency')
plt.plot(x[:-n_steps], predictions[:, 1], color='k', linestyle='--', label='Predicted Latency')
plt.title('Latency Prediction Comparison')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x[:-n_steps], scaled_cpu[:-n_steps], color='c', linestyle='-', label='Actual CPU Usage')
plt.plot(x[:-n_steps], predictions[:, 2], color='m', linestyle='--', label='Predicted CPU Usage')
plt.title('CPU Usage Prediction Comparison')
plt.legend()

plt.subplot(2,2, 4)
plt.plot(x[:-n_steps], scaled_memory[:-n_steps], color='y', linestyle='-', label='Actual Memory Usage')
plt.plot(x[:-n_steps], predictions[:, 3], color='purple', linestyle='--', label='Predicted Memory Usage')
plt.title('Memory Usage Prediction Comparison')
plt.legend()

plt.tight_layout()
plt.show()


# Display actual and predicted values near the end of the dataset
actual = data[950]  # Close to the end of the distribution
predicted = predictions[950 - n_steps]
print("Actual Values (Throughput, Latency, CPU, Memory):", actual)
print("Predicted Values (Throughput, Latency, CPU, Memory):", predicted)
