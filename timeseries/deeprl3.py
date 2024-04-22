import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

# Constants for the reward function
A, B, C, D, E = 1, 2, 3, 4, 5

# Load the data
data = pd.read_csv('system_metrics.csv')
time_steps = data['Time'].values
throughput = data['Throughput'].values
latency = data['Latency'].values
cpu_usage = data['CPU Usage'].values
memory_usage = data['Memory Usage'].values

# Initial number of VMs
initial_vms = 6

class Environment:
    def __init__(self, throughput, latency, cpu_usage, memory_usage):
        self.throughput = throughput
        self.latency = latency
        self.cpu_usage = cpu_usage
        self.memory_usage = memory_usage
        self.vm_counts = np.full(len(throughput), initial_vms)
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        self.vm_counts = np.full(len(self.throughput), initial_vms)
        return np.array([self.throughput[0], self.latency[0], self.vm_counts[0], self.cpu_usage[0], self.memory_usage[0]])

    def step(self, action):
        if action == 1 and self.vm_counts[self.current_step] > 1:
            self.vm_counts[self.current_step] -= 1  # Decrease VMs
        elif action == 2:
            self.vm_counts[self.current_step] += 1  # Increase VMs

        reward = (A * self.throughput[self.current_step] -
                  B * self.latency[self.current_step] -
                  C * self.vm_counts[self.current_step] -
                  D * self.cpu_usage[self.current_step] -
                  E * self.memory_usage[self.current_step])

        self.current_step += 1
        done = self.current_step >= len(self.throughput) - 1

        # Correcting how next_state is constructed
        next_state = np.array([
            self.throughput[min(self.current_step, len(self.throughput) - 1)],
            self.latency[min(self.current_step, len(self.latency) - 1)],
            self.vm_counts[min(self.current_step, len(self.vm_counts) - 1)],
            self.cpu_usage[min(self.current_step, len(self.cpu_usage) - 1)],
            self.memory_usage[min(self.current_step, len(self.memory_usage) - 1)]
        ])
        return next_state, reward, done



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(50, input_dim=self.state_size, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def act(self, state):
        state = state.reshape(1, -1)  # Ensure state has correct shape
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = state.reshape(1, -1)
            next_state = next_state.reshape(1, -1)
            target = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Initialize environment and agent
env = Environment(throughput, latency, cpu_usage, memory_usage)
agent = DQNAgent(state_size=5, action_size=3)  # Action size includes doing nothing as an option

# Training loop
episodes = 100
for e in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        if len(agent.memory) > 32:
            agent.replay(32)
        if done:
            print(f'Episode: {e+1}/{episodes}, Total reward: {total_reward}, Epsilon: {agent.epsilon:.2f}')
            break

# Plotting code goes here...



# Visualization and final thoughts
plt.figure(figsize=(12, 18))
# Add plotting code here similar to previous examples

plt.tight_layout()
plt.show()
