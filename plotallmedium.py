import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv('combined_metrics.csv')

# Convert the 'Time' column to a more readable format if necessary
data['Time'] = pd.to_datetime(data['Time'], unit='s')

# Setting the index to 'Time' for easier plotting and analysis
data.set_index('Time', inplace=True)

# Resample data by the index time and calculate the median if needed
# For example, if you have data recorded at irregular intervals and want to normalize:
# median_data = data.resample('S').median() # This example resamples data to every second

# If the data is already uniformly spaced and just needs grouping by exact times:
median_data = data.groupby(data.index).median()

# Plotting
fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)  # Adjust size as needed

# Plot Median of Average Throughput
axs[0].plot(median_data['Average Throughput'], label='Median Average Throughput', color='blue')
axs[0].set_ylabel('Throughput')
axs[0].legend()

# Plot Median of Average Latency
axs[1].plot(median_data['Average Latency'], label='Median Average Latency', color='red')
axs[1].set_ylabel('Latency (s)')
axs[1].legend()

# Plot Median of CPU Usage
axs[2].plot(median_data['CPU Usage %'], label='Median CPU Usage %', color='green')
axs[2].set_ylabel('CPU Usage (%)')
axs[2].legend()

# Plot Median of Memory Usage
axs[3].plot(median_data['Memory Usage %'], label='Median Memory Usage %', color='purple')
axs[3].set_ylabel('Memory Usage (%)')
axs[3].legend()

# Plot Median of Node Count
axs[4].plot(median_data['Node Count'], label='Median Node Count', color='orange')
axs[4].set_ylabel('Node Count')
axs[4].set_xlabel('Time')
axs[4].legend()

# Optional: Create additional plots for the logarithmic scales of CPU and Memory usage
# It might make sense to exclude or handle zeros before taking logs if zeros are possible in the data
fig2, axs2 = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Logarithmic plot for CPU Usage
axs2[0].plot(median_data['CPU Usage %'].replace(0, np.nan).dropna().apply(np.log), label='Log of Median CPU Usage %', color='green')
axs2[0].set_ylabel('Log CPU Usage (%)')
axs2[0].legend()

# Logarithmic plot for Memory Usage
axs2[1].plot(median_data['Memory Usage %'].replace(0, np.nan).dropna().apply(np.log), label='Log of Median Memory Usage %', color='purple')
axs2[1].set_ylabel('Log Memory Usage (%)')
axs2[1].set_xlabel('Time')
axs2[1].legend()

plt.show()

