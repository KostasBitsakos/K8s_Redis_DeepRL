import pandas as pd

# Load the dataset
df = pd.read_csv('system_metrics.csv')

# Multiply values as required
df['Throughput'] *= 10
df['Latency'] *= 10
df['CPU Usage'] *= 100
df['Memory Usage'] *= 100

# Save the updated data to a new file
df.to_csv('system_metrics_new.csv', index=False)
