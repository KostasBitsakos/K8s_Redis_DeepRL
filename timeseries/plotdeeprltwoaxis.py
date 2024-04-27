import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('updated6_system_metrics.csv')

# Setting the figure size and layout
fig, axes = plt.subplots(4, 1, figsize=(10, 15), sharex=True)

# Plotting each metric with Time on the x-axis and num_vms as dots
metrics = ['Throughput', 'Latency', 'CPU Usage', 'Memory Usage']
colors = ['blue', 'green', 'red', 'purple']  # Different color for each plot

for i, metric in enumerate(metrics):
    axes[i].plot(df['Time'], df[metric], label=metric, color=colors[i])
    axes[i].set_ylabel(metric)
    axes[i].legend(loc='upper right')

    # Create a twin y-axis for num_vms
    ax_vm = axes[i].twinx()
    ax_vm.scatter(df['Time'], df['num_vms'], color='black', label='num_vms', alpha=0.1)
    ax_vm.set_ylabel('Number of VMs')
    ax_vm.grid(False)  # Disable grid for the num_vms axis

# Set common labels
plt.xlabel('Time')
plt.tight_layout()

# Save the plots as PNG files
for i, metric in enumerate(metrics):
    plt.savefig(f'{metric}_plot.png')

plt.show()
