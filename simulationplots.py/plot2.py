import numpy as np
import matplotlib.pyplot as plt

# Parameters
x = np.linspace(0, 10, 1000)  # 1000 points from 0 to 10
amplitude = 1
frequency = 1
phase = 0

# Sinusoidal signal
y = amplitude * np.sin(2 * np.pi * frequency * x + phase)

# Adding Gaussian noise
noise = np.random.normal(0, 0.2, y.shape)
y_noisy = y + noise

# Adding a spike
spike_strength = 5  # The amplitude of the spike
spike_position = len(x) // 2  # Position in the middle of the array
y_noisy[spike_position] += spike_strength  # Add the spike to the noisy signal

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Original Sine Wave', color='blue')
plt.plot(x, y_noisy, label='Sine Wave with Noise and Spike', color='red', alpha=0.6)
plt.title('Sinusoidal Distribution with Noise and a Spike')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

