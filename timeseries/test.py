import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import math
import time

# Load the dataset
df = pd.read_csv('system_metrics.csv')

class Environment:
    def __init__(self, data):
        self.data = data
        self.num_vms = 6  # Starting number of VMs

    def load(self, t):
        # Directly use throughput as an indication of load
        return self.data.iloc[t]['Throughput']

    def latency(self, t):
        # Directly use latency from the dataset
        return self.data.iloc[t]['Latency']

    def capacity(self, t, vms):
        read_percentage = 0.75 + 0.25 * np.sin(2 * np.pi * t / 340)
        return 10 * vms * read_percentage

    def step(self, t, action):
        # Action: 0 decrease, 1 do nothing, 2 increase
        if action == 0 and self.num_vms > 1:
            self.num_vms -= 1
        elif action == 2 and self.num_vms < 20:
            self.num_vms += 1

        next_load = self.load(t + 1)
        next_capacity = self.capacity(t + 1, self.num_vms)
        reward = min(next_capacity, next_load) - 3 * self.num_vms
        done = t == len(self.data) - 2
        return self.num_vms, reward, done, next_load, self.latency(t + 1)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array([state]))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0]))
            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

def run():
    env = Environment(df)
    agent = DQNAgent(state_size=1, action_size=3)
    batch_size = 32
    result_data = []
    start_time = time.time()
    max_steps = len(df) - 1  # Set a maximum number of steps to the length of the dataset

    for t in range(max_steps):
        if t % 100 == 0:  # Print progress every 100 steps
            elapsed_time = time.time() - start_time
            print(f"Processing step {t+1}/{max_steps}. Time elapsed: {elapsed_time:.2f} seconds.")

        state = env.num_vms
        action = agent.act([state])
        next_state, reward, done, load, latency = env.step(t, action)
        agent.remember([state], action, reward, [next_state], done)
        state = next_state

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if done:
            print("Reached the end of the dataset.")
            break

        # Save to list
        new_row = {'Time': df.iloc[t]['Time'], 'Throughput': df.iloc[t]['Throughput'],
                   'Latency': df.iloc[t]['Latency'], 'CPU Usage': df.iloc[t]['CPU Usage'], 
                   'Memory Usage': df.iloc[t]['Memory Usage'], 'num_vms': env.num_vms}
        result_data.append(new_row)

    result_df = pd.DataFrame(result_data)
    result_df.to_csv('updated_system_metrics.csv', index=False)
    print("Simulation complete. Results saved to 'updated_system_metrics.csv'.")

if __name__ == '__main__':
    run()

