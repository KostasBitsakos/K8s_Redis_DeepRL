import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
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
        next_latency = self.latency(t + 1)

        # Reward function that considers load, capacity, and latency
        reward = 0
        if next_load > next_capacity:
            # Penalize if load exceeds capacity
            reward -= (next_load - next_capacity) * 2
        else:
            # Reward for having sufficient capacity
            reward += (next_capacity - next_load)

        # Penalize based on latency
        reward -= next_latency * 2

        # Adjust the reward based on the number of VMs
        reward -= self.num_vms

        done = t == len(self.data) - 2
        return self.num_vms, reward, done, next_load, next_latency

class DQNAgent:
    def __init__(self, state_size, action_size, sequence_length):
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(24, input_shape=(self.sequence_length, self.state_size), activation='relu', return_sequences=True))
        model.add(LSTM(12, activation='relu', return_sequences=False))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=self.learning_rate))
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
                # Reshape next_state to match the expected input shape of the model
                next_state = next_state.reshape((1, self.sequence_length, self.state_size))
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            # Reshape state to match the expected input shape of the model
            state = state.reshape((1, self.sequence_length, self.state_size))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def run():
    sequence_length = 16  # The sequence length for LSTM input
    state_size = 1  # Assuming a 1D state representation for simplicity
    action_size = 3  # Number of possible actions

    env = Environment(df)
    agent = DQNAgent(state_size, action_size, sequence_length)
    batch_size = 32
    result_data = []
    start_time = time.time()
    max_steps = len(df) - sequence_length  # Adjusted for sequence length

    for t in range(0, max_steps):
        state_sequence = np.array([env.load(i) for i in range(t, t + sequence_length)]).reshape((1, sequence_length, 1))
        action = agent.act(state_sequence)  # Pass 'state_sequence' directly to 'act'
        next_state_sequence = np.array([env.load(i) for i in range(t + 1, t + sequence_length + 1)]).reshape((1, sequence_length, 1))
        _, reward, done, load, latency = env.step(t, action)
        agent.remember(state_sequence, action, reward, next_state_sequence, done)

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
    result_df.to_csv('updated5_system_metrics.csv', index=False)
    print("Simulation complete. Results saved to 'updated_system_metrics.csv'.")
    print(f"Total reward accumulated over the simulation: {total_reward}")  # Print total reward

if __name__ == '__main__':
    run()

