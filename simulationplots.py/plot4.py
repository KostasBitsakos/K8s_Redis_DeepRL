import numpy as np
import matplotlib.pyplot as plt

# Parameters for the main sine wave
x = np.linspace(0, 10, 1000)  # 1000 points from 0 to 10
amplitude = 1
frequency = 1
phase = 0

# Generate the main sine wave
y = amplitude * np.sin(2 * np.pi * frequency * x + phase)

# Adding Gaussian noise
noise = np.random.normal(0, 0.2, y.shape)
y_noisy = y + noise

# Parameters for the sinusoidal spike
spike_amplitude = 3
spike_frequency = 10
spike_phase = 0
spike_length = 50  # length of the spike in number of samples

# Generate the spike as a sinusoidal wave
spike_x = np.linspace(0, 1, spike_length)
spike = spike_amplitude * np.sin(2 * np.pi * spike_frequency * spike_x + spike_phase)

# Apply threshold to keep only the positive parts of the spike
spike[spike < 0] = 0

# Position to insert the spike (around the middle)
spike_position = len(x) // 2

# Add the spike to the noisy signal
if spike_position + spike_length <= len(y_noisy):
    y_noisy[spike_position:spike_position + spike_length] += spike
else:
    y_noisy[spike_position:] += spike[:len(y_noisy) - spike_position]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Original Sine Wave', color='blue')
plt.plot(x, y_noisy, label='Sine Wave with Noise and Positive Sinusoidal Spike', color='red', alpha=0.6)
plt.title('Sinusoidal Distribution with Noise and a Positive Sinusoidal Spike')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

