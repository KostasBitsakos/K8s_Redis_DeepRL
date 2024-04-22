import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import random
import pandas as pd

# Constants for the reward function
A, B, C, D, E = 20, 2, 2, 2, 2

# Load the data
data = pd.read_csv('system_metrics.csv')
time_steps = data['Time'].values
throughput = data['Throughput'].values
latency = data['Latency'].values
cpu_usage = data['CPU Usage'].values
memory_usage = data['Memory Usage'].values

# Initial number of VMs
initial_vms = 6
vm_counts = np.full(len(time_steps), initial_vms)

# Define the RL environment and agent
class Environment:
    def __init__(self):
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return np.array([throughput[0], latency[0], vm_counts[0], cpu_usage[0], memory_usage[0]])

    def step(self, action):
        # Apply action to adjust VMs
        if action == 0 and vm_counts[self.current_step] > 1:
            vm_counts[self.current_step] -= 1  # Decrease VMs
        elif action == 1:
            vm_counts[self.current_step] += 1  # Increase VMs
        
        # Calculate reward
        reward = (A * throughput[self.current_step] -
                  B * latency[self.current_step] -
                  C * vm_counts[self.current_step] -
                  D * cpu_usage[self.current_step] -
                  E * memory_usage[self.current_step])

        self.current_step += 1
        done = self.current_step == len(time_steps) - 1
        next_state = np.array([throughput[self.current_step], latency[self.current_step], vm_counts[self.current_step], cpu_usage[self.current_step], memory_usage[self.current_step]])
        return next_state, reward, done

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

# Initialize environment and agent
env = Environment()
agent = DQNAgent(state_size=5, action_size=2)  # state: metrics and VM count, actions: increase or decrease VMs

# Run simulation
state = env.reset()
done = False
while not done:
    action = agent.act(state.reshape(1, -1))
    next_state, reward, done = env.step(action)
    state = next_state

# Plot results
plt.figure(figsize=(12, 18))
plt.subplot(5, 1, 1)
plt.plot(time_steps, throughput, label='Throughput')
plt.plot(time_steps, vm_counts * np.max(throughput) / max(vm_counts), label='VMs (scaled)', marker='o', linestyle='None', markersize=2)
plt.title('Throughput and VMs')
plt.legend()

plt.subplot(5, 1, 2)
plt.plot(time_steps, latency, label='Latency')
plt.plot(time_steps, vm_counts * np.max(latency) / max(vm_counts), label='VMs (scaled)', marker='o', linestyle='None', markersize=2)
plt.title('Latency and VMs')
plt.legend()

plt.subplot(5, 1, 3)
plt.plot(time_steps, cpu_usage, label='CPU Usage')
plt.plot(time_steps, vm_counts * np.max(cpu_usage) / max(vm_counts), label='VMs (scaled)', marker='o', linestyle='None', markersize=2)
plt.title('CPU Usage and VMs')
plt.legend()

plt.subplot(5, 1, 4)
plt.plot(time_steps, memory_usage, label='Memory Usage')
plt.plot(time_steps, vm_counts * np.max(memory_usage) / max(vm_counts), label='VMs (scaled)', marker='o', linestyle='None', markersize=2)
plt.title('Memory Usage and VMs')
plt.legend()

plt.tight_layout()
plt.show()
