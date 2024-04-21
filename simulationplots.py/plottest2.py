import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Load the model
model = load_model('my_lstm_model.keras')
print(tf.__version__)

# Set logger level
tf.get_logger().setLevel('DEBUG')

# Time vector for new distribution
x_new = np.linspace(0, 10, 1000)

# Parameters for base signal
amplitude_throughput_new = 0.95
frequency_throughput_new = 1.05
amplitude_latency_new = 0.75
frequency_latency_new = 1.95

# Generate sinusoidal base data
throughput_new = amplitude_throughput_new * np.sin(2 * np.pi * frequency_throughput_new * x_new) + 1.5
latency_new = amplitude_latency_new * np.sin(2 * np.pi * frequency_latency_new * x_new) + 1.2

# Add noise
throughput_noise_new = np.random.normal(0, 0.2, throughput_new.shape)
latency_noise_new = np.random.normal(0, 0.1, latency_new.shape)
throughput_new += throughput_noise_new
latency_new += latency_noise_new

# Introduce a spike in the middle of the data series
spike_length = 100  # Length of the spike
spike_position = 500  # Start of the spike
spike_amplitude = 3   # Amplitude of the spike

# Apply spike to throughput and latency
throughput_new[spike_position:spike_position + spike_length] += spike_amplitude * np.abs(np.sin(2 * np.pi * 10 * np.linspace(0, 1, spike_length)))
latency_new[spike_position:spike_position + spike_length] += spike_amplitude * 0.8 * np.abs(np.sin(2 * np.pi * 20 * np.linspace(0, 1, spike_length)))

# Normalize the data for model prediction
scaled_throughput_new = (throughput_new - np.min(throughput_new)) / (np.max(throughput_new) - np.min(throughput_new))
scaled_latency_new = (latency_new - np.min(latency_new)) / (np.max(latency_new) - np.min(latency_new))

# Reshape data for LSTM [Assuming n_steps and the LSTM input shape]
n_steps = 10
data_new = np.column_stack((scaled_throughput_new, scaled_latency_new))
input_seq = np.array([data_new[i:i+n_steps] for i in range(len(data_new) - n_steps)])

# Predicting with LSTM
predictions = model.predict(input_seq)

# Plotting
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(x_new[:-n_steps], scaled_throughput_new[:-n_steps], 'b', label='Actual Throughput')
plt.plot(x_new[:-n_steps], predictions[:, 0], 'r--', label='Predicted Throughput')
plt.title('Throughput Prediction Comparison')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_new[:-n_steps], scaled_latency_new[:-n_steps], 'g', label='Actual Latency')
plt.plot(x_new[:-n_steps], predictions[:, 1], 'k--', label='Predicted Latency')
plt.title('Latency Prediction Comparison')
plt.legend()

plt.show()

# Print actual vs predicted values
print("Actual Values (Throughput, Latency):", data_new[950])  # Close to the end of the distribution
print("Predicted Values (Throughput, Latency):", predictions[950 - n_steps])
