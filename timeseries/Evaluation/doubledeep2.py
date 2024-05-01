import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import time

# Define the environment
class Environment:
    def __init__(self):
        self.time = 0
        self.num_vms = 6  # Starting number of VMs
        self.load_amplitude = 50
        self.load_period = 250
        self.read_amplitude = 0.25
        self.read_period = 340
        self.cpu_range = (30, 50)
        self.memory_range = (40, 60)
        self.latency_amplitude = 0.1

    def load(self, t):
        return self.load_amplitude + self.load_amplitude * np.sin(2 * np.pi * t / self.load_period)

    def read_percentage(self, t):
        return 0.75 + self.read_amplitude * np.sin(2 * np.pi * t / self.read_period)

    def capacity(self, t):
        return 10 * self.num_vms * self.read_percentage(t)

    def cpu_usage(self, t):
        return np.random.uniform(*self.cpu_range)

    def memory_usage(self, t):
        return np.random.uniform(*self.memory_range)

    def latency(self, t):
        return self.latency_amplitude * np.sin(2 * np.pi * t / self.load_period)

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

        return self.num_vms, next_load, reward, self.cpu_usage(self.time), self.memory_usage(self.time), self.latency(self.time), self.read_percentage(self.time)


# Define the Double DQN Agent
class DDQNAgent:
    def __init__(self, state_size, action_size, sequence_length):
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
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
        state = np.reshape(state, (1, self.sequence_length, self.state_size))
        act_values = self.online_model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.array([next_state])
                next_state = np.reshape(next_state, (1, self.sequence_length, self.state_size))
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)[0]))
            state = np.array([state])
            state = np.reshape(state, (1, self.sequence_length, self.state_size))
            target_f = self.online_model.predict(state)
            target_f[0][action] = target
            self.online_model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.update_target_model()


def train_agent(train_steps, max_episodes_per_step=1000):
    sequence_length = 10  # The sequence length for LSTM input
    state_size = 7  # Throughput, Load, CPU Usage, Memory Usage, Latency, Number of VMs, Read Percentage
    action_size = 3  # Number of possible actions

    env = Environment()
    agent = DDQNAgent(state_size, action_size, sequence_length)  # Pass sequence_length here
    batch_size = 32
    result_data = []

    total_reward = 0
    episodes = 0
    for step in range(train_steps):
        state = np.zeros((sequence_length, state_size))
        for t in range(sequence_length):
            state[t][0] = env.load(env.time + t)
            state[t][1] = env.read_percentage(env.time + t)
            state[t][2] = env.cpu_usage(env.time + t)
            state[t][3] = env.memory_usage(env.time + t)
            state[t][4] = env.latency(env.time + t)
            state[t][5] = env.num_vms
            state[t][6] = env.read_percentage(env.time + t)

        done = False
        total_reward = 0
        while not done and episodes < max_episodes_per_step:
            action = agent.act(state)
            num_vms, next_load, reward, cpu_usage, memory_usage, latency, read_percentage = env.step(action)
            total_reward += reward

            next_state = np.zeros((sequence_length, state_size))
            next_state[:-1] = state[1:]
            next_state[-1][0] = next_load
            next_state[-1][1] = read_percentage
            next_state[-1][2] = cpu_usage
            next_state[-1][3] = memory_usage
            next_state[-1][4] = latency
            next_state[-1][5] = num_vms
            next_state[-1][6] = read_percentage

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if num_vms == 1 and next_load == 1:  # Terminal state condition
                done = True

            episodes += 1

        result_data.append({'Time': env.time, 'Throughput': state[-1][0], 'Latency': state[-1][4],
                            'CPU Usage': state[-1][2], 'Memory Usage': state[-1][3], 'num_vms': num_vms})

        if step % 100 == 0:
            print(f"Training step {step}/{train_steps}. Total reward: {total_reward}")

    result_df = pd.DataFrame(result_data)
    result_df.to_csv(f'training_output_{train_steps}.csv', index=False)
    print(f"Training completed for {train_steps} steps. Results saved to 'training_output_{train_steps}.csv'. Total reward: {total_reward}")


def evaluate_agent(eval_steps, model_file):
    sequence_length = 10  # The sequence length for LSTM input
    state_size = 7  # Throughput, Load, CPU Usage, Memory Usage, Latency, Number of VMs, Read Percentage
    action_size = 3  # Number of possible actions

    env = Environment()
    agent = DDQNAgent(state_size, action_size, sequence_length)  # Pass sequence_length here
    batch_size = 32
    result_data = []

    # Load the model
    agent.online_model.load_weights(model_file)
    agent.update_target_model()

    total_reward = 0
    for _ in range(eval_steps):  # Run for the specified number of evaluation steps
        state = np.zeros((sequence_length, state_size))
        for t in range(sequence_length):
            state[t][0] = env.load(env.time + t)
            state[t][1] = env.read_percentage(env.time + t)
            state[t][2] = env.cpu_usage(env.time + t)
            state[t][3] = env.memory_usage(env.time + t)
            state[t][4] = env.latency(env.time + t)
            state[t][5] = env.num_vms
            state[t][6] = env.read_percentage(env.time + t)

        done = False
        while not done:
            action = agent.act(state)
            num_vms, next_load, reward, cpu_usage, memory_usage, latency, read_percentage = env.step(action)
            total_reward += reward

            next_state = np.zeros((sequence_length, state_size))
            next_state[:-1] = state[1:]
            next_state[-1][0] = next_load
            next_state[-1][1] = read_percentage
            next_state[-1][2] = cpu_usage
            next_state[-1][3] = memory_usage
            next_state[-1][4] = latency
            next_state[-1][5] = num_vms
            next_state[-1][6] = read_percentage

            state = next_state

            if num_vms == 1 and next_load == 1:  # Terminal state condition
                done = True

        result_data.append({'Time': env.time, 'Throughput': state[-1][0], 'Latency': state[-1][4],
                            'CPU Usage': state[-1][2], 'Memory Usage': state[-1][3], 'num_vms': num_vms})

    result_df = pd.DataFrame(result_data)
    result_df.to_csv(f'evaluation_output_{eval_steps}_{model_file}.csv', index=False)
    print(f"Evaluation completed using model '{model_file}'. Results saved to 'evaluation_output_{eval_steps}_{model_file}.csv'. Total reward: {total_reward}")


if __name__ == '__main__':
    train_steps_list = [2000, 5000, 10000, 20000, 500000]
    eval_steps = 2000

    for train_steps in train_steps_list:
        print(f"Training for {train_steps} steps...")
        train_agent(train_steps)
        print("Training completed.\n")

    print(f"Evaluating the trained agents for {eval_steps} steps...")
    for train_steps in train_steps_list:
        model_file = f'training_output_{train_steps}.h5'
        evaluate_agent(eval_steps, model_file)
