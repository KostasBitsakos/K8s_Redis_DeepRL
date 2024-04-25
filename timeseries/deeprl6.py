import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

# Load the dataset
df = pd.read_csv('system_metrics.csv')

# Initialize a new column for the number of VMs
df['num_vms'] = 6  # Starting number of VMs

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
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def run():
    state_size = 5
    action_size = 3  # Now we have 3 actions: 0 (decrease), 1 (maintain), 2 (increase) VMs
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episodes = 10  # Control the number of episodes

    for episode in range(episodes):
        total_reward = 0
        for index, row in df.iterrows():
            state = np.array([[row['Throughput'], row['Latency'], row['CPU Usage'], row['Memory Usage'], row['num_vms']]])
            action = agent.act(state)
            # Apply the action to get the next state
            next_num_vms = row['num_vms'] + (action - 1)  # Subtract 1 to map actions to changes
            next_state = np.array([[row['Throughput'], row['Latency'], row['CPU Usage'], row['Memory Usage'], next_num_vms]])
            reward = 20*row['Throughput'] - row['Latency'] - 10*row['CPU Usage'] - 10*row['Memory Usage'] - 5*next_num_vms
            done = index == df.index[-1]
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            # Update and print status
            df.loc[index, 'num_vms'] = next_num_vms  # Update num_vms based on action
            print(f"Episode {episode+1}, Step {index+1}, Total Reward: {total_reward}")

        if episode % 1 == 0:  # Save every episode
            agent.model.save(f'model_episode_{episode}.keras')
            df.to_csv('updated4_system_metrics.csv', index=False)

if __name__ == '__main__':
    run()
