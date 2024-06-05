import numpy as np
import matplotlib.pyplot as plt


class Environment:
    def __init__(self):
        self.time = 0
        self.num_vms = 6  # Starting number of VMs
        self.load_amplitude = 50
        self.load_period = 250
        self.read_amplitude = 0.60  # Increased read percentage amplitude
        self.read_period = 340
        self.cpu_range = (30, 50)
        self.memory_range = (40, 60)
        self.latency_amplitude = 10

    def load(self, t):
        base_load = self.load_amplitude + self.load_amplitude * np.sin(2 * np.pi * t / self.load_period)
        spike_time = 500  # Adjust the spike time as needed
        spike_amplitude = 100  # Adjust the spike amplitude as needed
        if t == spike_time:
            return base_load + spike_amplitude
        else:
            return base_load

    def read_percentage(self, t):
        base_read = 0.75 + self.read_amplitude * np.sin(2 * np.pi * t / self.read_period)
        spike_time = 700  # Adjust the spike time as needed
        spike_amplitude = 0.3  # Adjust the spike amplitude as needed
        if t == spike_time:
            return base_read + spike_amplitude
        else:
            return base_read

    def capacity(self, t):
        return 10 * self.num_vms * self.read_percentage(t)

    def cpu_usage(self, t):
        return np.random.uniform(*self.cpu_range)

    def memory_usage(self, t):
        return np.random.uniform(*self.memory_range)

    def latency(self, t):
        load = self.load(t)
        delay = 10  # Adjust the delay as needed
        return load * 0.1 * np.sin(2 * np.pi * (t - delay) / self.load_period)

    def step(self, action):
        # Action: 0 decrease, 1 do nothing, 2 increase
        if action == 0 and self.num_vms > 1:
            self.num_vms -= 1
        elif action == 2 and self.num_vms < 20:
            self.num_vms += 1

        self.time += 1
        next_load = self.load(self.time)
        next_capacity = self.capacity(self.time)
        reward = min(next_capacity, next_load) - 3 * self.num_vms

        return self.num_vms, next_load, reward


if __name__ == "__main__":
    env = Environment()

    # Plotting
    time_steps = range(5000)
    env_load = [env.load(t) for t in time_steps]
    env_capacity = [env.capacity(t) for t in time_steps]
    env_read = [env.read_percentage(t) for t in time_steps]

    plt.plot(time_steps, env_load, label='Load')
    plt.plot(time_steps, env_capacity, label='Capacity')
    plt.plot(time_steps, env_read, label='Read Percentage')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig('load.png')
    plt.show()
