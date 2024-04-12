import numpy as np
import matplotlib.pyplot as plt

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

# Plotting
plt.figure(figsize=(10, 5))

# Normalize throughput and latency for better visualization on the same scale
scaled_throughput = (throughput - np.min(throughput)) / (np.max(throughput) - np.min(throughput))
scaled_latency = (latency - np.min(latency)) / (np.max(latency) - np.min(latency))

# Plot throughput and latency on the same plot
plt.plot(x, scaled_throughput, label='Normalized Throughput', color='blue')
plt.plot(x, scaled_latency, label='Normalized Latency', color='red')
plt.title('Throughput and Latency Simulation for Redis Cluster')
plt.xlabel('Time')
plt.ylabel('Normalized Metrics')
plt.legend()

plt.show()

