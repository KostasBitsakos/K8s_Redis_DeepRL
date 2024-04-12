import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
data = pd.read_csv('metrics3.csv')

# Convert timestamp to a more readable format, if necessary
# For simplicity, we'll assume it's already in a human-readable form

# Plotting throughput over time
plt.figure(figsize=(12, 6))
plt.plot(data['timestamp'], data['throughput'], label='Throughput (requests/sec)', color='blue', marker='o')

# Plotting average latency over time on a secondary y-axis
plt.ylabel('Throughput (requests/sec)')
plt.xlabel('Time')
plt.twinx()
plt.plot(data['timestamp'], data['average_latency'], label='Average Latency (sec)', color='red', marker='x')
plt.ylabel('Average Latency (sec)')

plt.title('Redis Metrics Over Time')
plt.legend()
plt.show()

