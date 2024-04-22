import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Time vector
x = np.linspace(0, 10, 1000)

# Throughput simulation parameters
amplitude_throughput = 1
frequency_throughput = 1

# Generate sinusoidal throughput
throughput = amplitude_throughput * np.sin(2 * np.pi * frequency_throughput * x) + 1.5

# Adding Gaussian noise to throughput
throughput_noise = np.random.normal(0, 0.2, throughput.shape)
throughput += throughput_noise

# Introduce a spike in throughput
spike_length = 100
spike_position = 500
spike_amplitude = 3
spike_throughput = spike_amplitude * np.abs(np.sin(2 * np.pi * 10 * np.linspace(0, 1, spike_length)))
throughput[spike_position:spike_position + spike_length] += spike_throughput

# Simulate Latency: add a lag to the response based on throughput
lag = 10  # 10 timesteps lag
latency = np.roll(throughput * 0.8 + np.random.normal(0, 0.1, throughput.shape), lag)
latency[:lag] = throughput[:lag] * 0.8  # Handle the wrap-around effect

# Simulate CPU and Memory usage: delayed response to throughput changes
cpu_usage = np.roll(0.3 + 0.1 * throughput / np.max(throughput) + 0.02 * np.sin(2 * np.pi * 0.3 * x), lag)
cpu_usage[:lag] = 0.3 + 0.1 * throughput[:lag] / np.max(throughput)

memory_usage = np.roll(0.4 + 0.15 * throughput / np.max(throughput) + 0.03 * np.sin(2 * np.pi * 0.4 * x), lag)
memory_usage[:lag] = 0.4 + 0.15 * throughput[:lag] / np.max(throughput)

# Create a DataFrame and save all metrics
data = pd.DataFrame({
    'Time': x,
    'Throughput': throughput,
    'Latency': latency,
    'CPU Usage': cpu_usage,
    'Memory Usage': memory_usage
})

# Save data to CSV
data.to_csv('system_metrics.csv', index=False)

# Normalize CPU, Memory, Throughput, and Latency for plotting
scaled_cpu = (cpu_usage - np.min(cpu_usage)) / (np.max(cpu_usage) - np.min(cpu_usage))
scaled_memory = (memory_usage - np.min(memory_usage)) / (np.max(memory_usage) - np.min(memory_usage))
scaled_throughput = (throughput - np.min(throughput)) / (np.max(throughput) - np.min(throughput))
scaled_latency = (latency - np.min(latency)) / (np.max(latency) - np.min(latency))

# Plotting each metric on a different subplot
plt.figure(figsize=(10, 12))

plt.subplot(4, 1, 1)
plt.plot(x, scaled_throughput, label='Normalized Throughput', color='blue')
plt.title('Normalized Throughput')
plt.xlabel('Time')
plt.ylabel('Throughput')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(x, scaled_latency, label='Normalized Latency', color='red')
plt.title('Normalized Latency')
plt.xlabel('Time')
plt.ylabel('Latency')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(x, scaled_cpu, label='Normalized CPU Usage', color='green')
plt.title('Normalized CPU Usage')
plt.xlabel('Time')
plt.ylabel('CPU Usage')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(x, scaled_memory, label='Normalized Memory Usage', color='purple')
plt.title('Normalized Memory Usage')
plt.xlabel('Time')
plt.ylabel('Memory Usage')
plt.legend()

plt.tight_layout()
plt.show()
