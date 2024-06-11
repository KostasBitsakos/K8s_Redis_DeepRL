import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self):
        self.time = 0
        self.num_vms = 6  # Starting number of VMs
        self.load_amplitude = 1
        self.load_period = 1000 / (2 * np.pi)
        self.read_amplitude = 0.8  # Increased read percentage amplitude
        self.read_period = 1000 / (4 * np.pi)
        self.cpu_range = (30, 50)
        self.memory_range = (40, 60)
        self.latency_amplitude = 1.2
        self.spike_position = 1500  # Moved spike occurrence later
        self.spike_length = 100  # Spike duration
        self.spike_amplitude = 3  # Spike amplitude

    def load(self, t):
        # Generate sinusoidal load with noise
        load = self.load_amplitude * np.sin(2 * np.pi * t / self.load_period) + 1.5
        load += np.random.normal(0, 0.2)
        # Introduce a spike in load
        if self.spike_position <= t < self.spike_position + self.spike_length:
            load += self.spike_amplitude * np.abs(np.sin(2 * np.pi * 10 * (t - self.spike_position) / self.spike_length))
        return load

    def read_percentage(self, t):
        # Generate sinusoidal read percentage with noise
        read = self.read_amplitude * np.sin(2 * np.pi * t / self.read_period) + 1.2
        read += np.random.normal(0, 0.1)
        # Introduce a spike in read percentage
        if self.spike_position <= t < self.spike_position + self.spike_length:
            read += self.spike_amplitude * 0.3 * np.abs(np.sin(2 * np.pi * 20 * (t - self.spike_position) / self.spike_length))
        return read

    def capacity(self, t):
        # Generate base capacity
        capacity = 10 * self.num_vms * self.read_percentage(t)
        # Introduce a spike in capacity
        if self.spike_position <= t < self.spike_position + self.spike_length:
            capacity += self.spike_amplitude * 0.5 * np.abs(np.sin(2 * np.pi * 20 * (t - self.spike_position) / self.spike_length))
        return capacity

    def cpu_usage(self, t):
        # Generate random CPU usage within range
        cpu_usage = np.random.uniform(*self.cpu_range)
        # Introduce a spike in CPU usage
        if self.spike_position <= t < self.spike_position + self.spike_length:
            cpu_usage += self.spike_amplitude
        return cpu_usage

    def memory_usage(self, t):
        # Generate random memory usage within range
        memory_usage = np.random.uniform(*self.memory_range)
        # Introduce a spike in memory usage
        if self.spike_position <= t < self.spike_position + self.spike_length:
            memory_usage += self.spike_amplitude
        return memory_usage

    def latency(self, t):
        # Generate sinusoidal latency with noise
        latency = self.latency_amplitude * np.sin(2 * np.pi * 2 * t / self.load_period) + 1.2
        latency += np.random.normal(0, 0.1)
        # Introduce a spike in latency
        if self.spike_position <= t < self.spike_position + self.spike_length:
            latency += self.spike_amplitude * 0.8 * np.abs(np.sin(2 * np.pi * 20 * (t - self.spike_position) / self.spike_length))
        return latency
    def step(self, action):
    # Action: 0 decrease, 1 do nothing, 2 increase
        if action == 0 and self.num_vms > 1:
            self.num_vms -= 1
        elif action == 2 and self.num_vms < 20:
            self.num_vms += 1

        self.time += 1
        next_load = self.load(self.time)
        next_capacity = self.capacity(self.time)
        next_cpu_usage = self.cpu_usage(self.time)
        next_memory_usage = self.memory_usage(self.time)
        next_latency = self.latency(self.time)

    # Calculate the reward
        reward = min(next_capacity, next_load) - 3 * self.num_vms

    # Assemble the next state
        next_state = np.array([next_load, next_capacity, next_cpu_usage, next_memory_usage, next_latency])

    # Check if the episode is done
        done = self.time >= 5000  # or some other condition that defines an end of an episode

        return next_state, reward, done
    def reset(self):
    # Reset the environment state to the initial conditions
        self.time = 0
        self.num_vms = 6  # Reset the number of VMs to the initial value
    # You may also want to reset any other variables that should be reinitialized at the start of an episode
    # Return the initial state of the environment
        initial_load = self.load(self.time)
        initial_capacity = self.capacity(self.time)
        initial_cpu_usage = self.cpu_usage(self.time)
        initial_memory_usage = self.memory_usage(self.time)
        initial_latency = self.latency(self.time)
        initial_state = np.array([initial_load, initial_capacity, initial_cpu_usage, initial_memory_usage, initial_latency])
        return initial_state




if __name__ == "__main__":
    env = Environment()

    # Plotting
    time_steps = range(5000)
    env_load = [env.load(t) for t in time_steps]
    env_capacity = [env.capacity(t) for t in time_steps]
    env_read = [env.read_percentage(t) for t in time_steps]
    env_cpu = [env.cpu_usage(t) for t in time_steps]
    env_memory = [env.memory_usage(t) for t in time_steps]
    env_latency = [env.latency(t) for t in time_steps]

    plt.figure(figsize=(10, 8))

    plt.subplot(3, 2, 1)
    plt.plot(time_steps, env_load, label='Load')
    plt.xlabel('Time')
    plt.ylabel('Load')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(time_steps, env_capacity, label='Capacity')
    plt.xlabel('Time')
    plt.ylabel('Capacity')
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(time_steps, env_read, label='Read Percentage')
    plt.xlabel('Time')
    plt.ylabel('Read Percentage')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(time_steps, env_cpu, label='CPU Usage')
    plt.xlabel('Time')
    plt.ylabel('CPU Usage')
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(time_steps, env_memory, label='Memory Usage')
    plt.xlabel('Time')
    plt.ylabel('Memory Usage')
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(time_steps, env_latency, label='Latency')
    plt.xlabel('Time')
    plt.ylabel('Latency')
    plt.legend()

    plt.tight_layout()
    plt.savefig('metrics.png')
    plt.show()
