import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Input

from tensorflow.keras.optimizers import Adam
import tensorflow as tf
tf.get_logger().setLevel('DEBUG')
model = load_model('my_lstm_model.keras', custom_objects=None, compile=True, safe_mode=True)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')





# Time vector for new distribution
x_new = np.linspace(0, 10, 1000)

# Slightly adjusted parameters
amplitude_throughput_new = 0.95  # Slightly lower amplitude
frequency_throughput_new = 1.05  # Slightly higher frequency

# Generate new sinusoidal throughput with minor changes
throughput_new = amplitude_throughput_new * np.sin(2 * np.pi * frequency_throughput_new * x_new) + 1.5
throughput_noise_new = np.random.normal(0, 0.2, throughput_new.shape)
throughput_new += throughput_noise_new

# Adjusted latency simulation
frequency_latency_new = 1.95
latency_new = amplitude_throughput_new * 0.75 * np.sin(2 * np.pi * frequency_latency_new * x_new) + 1.2
latency_new += np.random.normal(0, 0.1, latency_new.shape)

# Normalize the new data for model prediction
scaled_throughput_new = (throughput_new - np.min(throughput_new)) / (np.max(throughput_new) - np.min(throughput_new))
scaled_latency_new = (latency_new - np.min(latency_new)) / (np.max(latency_new) - np.min(latency_new))

# Reshape data for LSTM [Assuming n_steps and the LSTM input shape]
n_steps = 10  # Example window size used during training
features = 2
data_new = np.column_stack((scaled_throughput_new, scaled_latency_new))
input_seq = np.array([data_new[i:i+n_steps] for i in range(len(data_new) - n_steps)])

# Predicting with LSTM
predictions = model.predict(input_seq)

# Select a specific timestep to compare and display
timestep_to_display = 950  # Close to the end of the distribution
actual = data_new[timestep_to_display]
predicted = predictions[timestep_to_display - n_steps]

# Plot the results
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
print("Actual Values (Throughput, Latency):", actual)
print("Predicted Values (Throughput, Latency):", predicted)
