import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV data into a DataFrame
df = pd.read_csv('metrics.csv')

# Convert the 'Time' column to datetime format for better plotting
df['Time'] = pd.to_datetime(df['Time'], unit='s')

# Set the 'Time' column as the index of the DataFrame
df.set_index('Time', inplace=True)

# Ensure all values are positive for log transformation
# Assuming 'Average Latency', 'CPU Usage %', 'Memory Usage %' are always positive

# Calculate the log of the metrics
df['Log Average Latency'] = np.log(df['Average Latency'] + 1)  # Add 1 to avoid log(0)
df['Log CPU Usage %'] = np.log(df['CPU Usage %'] + 1)
df['Log Memory Usage %'] = np.log(df['Memory Usage %'] + 1)

# Plot for Latency and Throughput, including log latency
plt.figure(figsize=(12, 6))
plt.xlabel('Time')
plt.ylabel('Value / Log Value')
plt.plot(df.index, df['Average Throughput'], label='Average Throughput (requests/sec)', color='blue', marker='o', linestyle='-')
plt.plot(df.index, df['Average Latency'], label='Average Latency (sec)', color='orange', marker='x', linestyle='-.')
plt.plot(df.index, df['Log Average Latency'], label='Log Average Latency', color='red', marker='x', linestyle='--')
plt.title('Latency and Throughput Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot for CPU and Memory Usage, including their log transformations
plt.figure(figsize=(12, 6))
plt.xlabel('Time')
plt.ylabel('Usage Percentage / Log Usage Percentage')
plt.plot(df.index, df['CPU Usage %'], label='CPU Usage (%)', color='green', marker='s', linestyle='-')
plt.plot(df.index, df['Memory Usage %'], label='Memory Usage (%)', color='purple', marker='^', linestyle='-.')
plt.plot(df.index, df['Log CPU Usage %'], label='Log CPU Usage (%)', color='lightgreen', marker='s', linestyle='--')
plt.plot(df.index, df['Log Memory Usage %'], label='Log Memory Usage (%)', color='violet', marker='^', linestyle=':')
plt.title('CPU and Memory Usage Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

