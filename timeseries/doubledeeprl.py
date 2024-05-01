import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import time

# Load the dataset
df = pd.read_csv('system_metrics_new.csv')

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
        return 5 * vms * read_percentage

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

        # Adjust the reward based on the number of VMs and throughput
        if next_load < 0.5:  # Example threshold for low throughput
            reward += 0.1 * (1 / next_load)  # Increase reward for low throughput

        done = t == len(self.data) - 2
        return self.num_vms, reward, done, next_load, next_latency

class DDQNAgent:
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
        self.online_model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=(self.sequence_length, self.state_size), activation='relu', return_sequences=True))
        model.add(LSTM(128, activation='relu', return_sequences=True))
        model.add(LSTM(64, activation='relu', return_sequences=False))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.online_model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.array(state)
        state = np.reshape(state, (state.shape[0], self.sequence_length, self.state_size))
        act_values = self.online_model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.array([next_state])
                next_state = np.reshape(next_state, (next_state.shape[0], self.sequence_length, self.state_size))
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)[0]))
            state = np.array([state])
            state = np.reshape(state, (state.shape[0], self.sequence_length, self.state_size))
            target_f = self.online_model.predict(state)
            target_f[0][action] = target
            self.online_model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.update_target_model()


def run():
    sequence_length = 16  # The sequence length for LSTM input
    state_size = 1  # Assuming a 1D state representation for simplicity
    action_size = 3  # Number of possible actions

    env = Environment(df)
    agent = DDQNAgent(state_size, action_size, sequence_length)  # Pass sequence_length here
    batch_size = 32
    result_data = []
    start_time = time.time()
    max_steps = len(df) - sequence_length  # Adjusted for sequence length
    total_reward = 0  # Initialize total reward accumulator

    for t in range(0, max_steps):
        if t % 100 == 0:  # Print progress every 100 steps
            elapsed_time = time.time() - start_time
            print(f"Processing step {t+1}/{max_steps}. Time elapsed: {elapsed_time:.2f} seconds.")

        # Prepare input data sequence for LSTM
        state_sequence = np.array([env.load(i) for i in range(t, t + sequence_length)]).reshape((1, sequence_length, state_size))

        # Pass the state sequence directly to the agent's act method
        action = agent.act(state_sequence)

        # Prepare next state sequence for LSTM
        next_state_sequence = np.array([env.load(i) for i in range(t + 1, t + sequence_length + 1)]).reshape((1, sequence_length, state_size))

        # Perform the step and update the agent's memory
        num_vms, reward, done, load, latency = env.step(t, action)
        total_reward += reward  # Accumulate total reward
        agent.remember(state_sequence, action, reward, next_state_sequence, done)

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if done:
            print("Reached the end of the dataset.")
            break

        # Save to list
        new_row = {'Time': df.iloc[t]['Time'], 'Throughput': df.iloc[t]['Throughput'],
                   'Latency': df.iloc[t]['Latency'], 'CPU Usage': df.iloc[t]['CPU Usage'], 
                   'Memory Usage': df.iloc[t]['Memory Usage'], 'num_vms': num_vms}
        result_data.append(new_row)

    result_df = pd.DataFrame(result_data)
    result_df.to_csv('updated7_system_metrics.csv', index=False)
    print("Simulation complete. Results saved to 'updated_system_metrics.csv'.")
    print(f"Total reward accumulated over the simulation: {total_reward}")  # Print total reward

if __name__ == '__main__':
    run()
