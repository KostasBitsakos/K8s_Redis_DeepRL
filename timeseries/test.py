import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate a time range of daily intervals
num_days = 365
date_rng = pd.date_range(start='1/1/2022', end='31/12/2022', freq='D')
df = pd.DataFrame(date_rng, columns=['date'])
df.set_index('date', inplace=True)

# Generate the base sinusoidal time series
amplitude = 1
frequency = 2 * np.pi / 30  # period of 30 days
df['sinusoidal'] = amplitude * np.sin(frequency * np.arange(num_days))

# Add noise to the base sinusoidal data
base_noise_level = 0.1  # Adjust this parameter to increase/decrease the noise level
base_noise = np.random.normal(0, base_noise_level, num_days)
df['sinusoidal'] += base_noise

# Create a spike using the positive part of a sinusoidal
spike_start = 150  # Start day of the spike
spike_end = 200    # End day of the spike, exclusive
spike_frequency = 2 * np.pi / 10  # shorter period for the spike
spike_amplitude = 3
spike_length = spike_end - spike_start  # Calculate length of spike period
spike_values = spike_amplitude * np.sin(spike_frequency * np.arange(spike_length))
spike_values[spike_values < 0] = 0  # Only take the positive part of the sine wave

# Add noise to the spike
spike_noise_level = 0.5  # Noise level for the spike
spike_noise = np.random.normal(0, spike_noise_level, spike_length)
spike_values += spike_noise

# Ensure the length of spike_values matches the spike period in the DataFrame
assert len(spike_values) == (spike_end - spike_start), "Spike length mismatch!"

# Add the spike to the base sinusoidal values
df.loc[df.index[spike_start:spike_end], 'sinusoidal'] += spike_values

# Plot the resulting time series
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['sinusoidal'], label='Sinusoidal with Spike and Noise')
plt.title('Time Series with Sinusoidal Distribution, Spike, and Noise')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
