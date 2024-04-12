import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data into a DataFrame
df = pd.read_csv('metrics.csv')

# Convert the 'Time' column to datetime format for better plotting
df['Time'] = pd.to_datetime(df['Time'], unit='s')

# Set the 'Time' column as the index of the DataFrame
df.set_index('Time', inplace=True)

# Plotting the metrics
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Time')
#ax1.set_ylabel('Total Requests', color=color)
#ax1.plot(df.index, df['Total Requests'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Average Throughput, Average Latency, CPU Usage %, Memory Usage %', color=color)
ax2.plot(df.index, df['Average Throughput'], color='blue', label='Average Throughput')
ax2.plot(df.index, df['Average Latency'], color='green', label='Average Latency')
ax2.plot(df.index, df['CPU Usage %'], color='orange', label='CPU Usage %')
ax2.plot(df.index, df['Memory Usage %'], color='purple', label='Memory Usage %')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.legend(loc='upper left')
plt.show()

