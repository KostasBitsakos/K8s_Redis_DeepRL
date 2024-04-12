import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv('combined_metrics.csv')

# Convert the 'Time' column to a more readable format if necessary
data['Time'] = pd.to_datetime(data['Time'], unit='s')

# Setting the index to 'Time' for easier plotting
data.set_index('Time', inplace=True)

# Plotting
fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)  # Adjust size as needed

# Plot Average Throughput
axs[0].plot(data['Average Throughput'], label='Average Throughput', color='blue')
axs[0].set_ylabel('Throughput')
axs[0].legend()

# Plot Average Latency
axs[1].plot(data['Average Latency'], label='Average Latency', color='red')
axs[1].set_ylabel('Latency (s)')
axs[1].legend()

# Plot CPU Usage
axs[2].plot(data['CPU Usage %'], label='CPU Usage %', color='green')
axs[2].set_ylabel('CPU Usage (%)')
axs[2].legend()

# Plot Memory Usage
axs[3].plot(data['Memory Usage %'], label='Memory Usage %', color='purple')
axs[3].set_ylabel('Memory Usage (%)')
axs[3].legend()

# Plot Node Count
axs[4].plot(data['Node Count'], label='Node Count', color='orange')
axs[4].set_ylabel('Node Count')
axs[4].set_xlabel('Time')
axs[4].legend()

# Create second figure for logarithmic plots
fig2, axs2 = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Logarithmic plot for CPU Usage
axs2[0].plot(data['CPU Usage %'].replace(0, np.nan).dropna().apply(np.log), label='Log of CPU Usage %', color='green')
axs2[0].set_ylabel('Log CPU Usage (%)')
axs2[0].legend()

# Logarithmic plot for Memory Usage
axs2[1].plot(data['Memory Usage %'].replace(0, np.nan).dropna().apply(np.log), label='Log of Memory Usage %', color='purple')
axs2[1].set_ylabel('Log Memory Usage (%)')
axs2[1].set_xlabel('Time')
axs2[1].legend()

plt.show()

