import warnings 
import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

# Define the environment
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Environment:
    def __init__(self):
        self.time = 0
        self.num_vms = 6  # Starting number of VMs
        self.load_amplitude = 50
        self.load_period = 250
        self.read_amplitude = 0.30
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
        spike_amplitude = 0.5  # Adjust the spike amplitude as needed
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



# Define the Double DQN Agent
class DDQNAgent:
    def __init__(self, state_size, action_size, sequence_length):
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.memory = deque(maxlen=500)
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

    def act(self, state, verbose=False):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.array(state)
        state = np.reshape(state, (1, self.sequence_length, self.state_size))
        act_values = self.online_model.predict(state, verbose=verbose)
        return np.argmax(act_values[0])


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.array([next_state])
                next_state = np.reshape(next_state, (1, self.sequence_length, self.state_size))
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0]))

            state = np.array([state])
            state = np.reshape(state, (1, self.sequence_length, self.state_size))
            target_f = self.online_model.predict(state)
            target_f[0][action] = target
            self.online_model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.update_target_model()

def train_agent(train_steps, max_episodes_per_step=1000):
    env = Environment()
    state_size = 5
    action_size = 3
    sequence_length = 10
    batch_size = 16
    agent = DDQNAgent(state_size, action_size, sequence_length)

    rewards = []

    for _ in range(train_steps):
        state = np.zeros((sequence_length, state_size))
        total_reward = 0
        no_improvement_counter = 0
        best_reward = -np.inf
        
        done = False  # Initialize done outside the loop
        step_count = 0  # Initialize step_count

        while not done:
            for _ in range(max_episodes_per_step):
                action = agent.act(state, verbose=0)

                num_vms, next_load, reward = env.step(action)
                total_reward += reward

                next_state = np.zeros((sequence_length, state_size))
                next_state[:-1] = state[1:]
                next_state[-1][0] = next_load
                next_state[-1][1] = env.read_percentage(env.time + sequence_length)
                next_state[-1][2] = env.cpu_usage(env.time + sequence_length)
                next_state[-1][3] = env.memory_usage(env.time + sequence_length)
                next_state[-1][4] = env.latency(env.time + sequence_length)

                agent.remember(state, action, reward, next_state, done)
                state = next_state

                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
                if total_reward > best_reward:
                    best_reward = total_reward
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 1

                if no_improvement_counter > 100:  # Terminate if no improvement in 100 consecutive episodes
                    done = True
                    break  # Exit the loop when termination condition is met
                step_count += 1
                if step_count >= 1000:  # Terminate after 2000 steps
                    print("done")
                    done = True
                    break  # Exit the loop when termination condition is met

            rewards.append(total_reward)
            if done:
                break  # Exit the outer loop when termination condition is met

    with open(f"rewards_{train_steps}.txt", "w") as f:
        for reward in rewards:
            f.write(str(reward) + "\n")

    agent.online_model.save_weights(f'model_{train_steps}.weights.h5')
    print(f"Model for {train_steps} training steps saved to 'model_{train_steps}.h5'.")
    print("Training completed.\n")


def evaluate_agent(eval_steps, max_episodes_per_step=1000):
    env = Environment()
    state_size = 5
    action_size = 3
    sequence_length = 10
    batch_size = 16
    agent = DDQNAgent(state_size, action_size, sequence_length)

    agent.online_model.load_weights(f'model_{eval_steps}.weights.h5')

    print(f"Model for evaluation loaded from 'model_{eval_steps}.h5'.")

    result_data = []
    
    done = False  # Initialize done outside the loop

    for _ in range(eval_steps):
        state = np.zeros((sequence_length, state_size))
        total_reward = 0

        for _ in range(max_episodes_per_step):
            step_count = 0  # Initialize step_count

            while not done:
                action = agent.act(state, verbose=0)

                num_vms, next_load, reward = env.step(action)
                total_reward += reward

                next_state = np.zeros((sequence_length, state_size))
                next_state[:-1] = state[1:]
                next_state[-1][0] = next_load
                next_state[-1][1] = env.read_percentage(env.time + sequence_length)
                next_state[-1][2] = env.cpu_usage(env.time + sequence_length)
                next_state[-1][3] = env.memory_usage(env.time + sequence_length)
                next_state[-1][4] = env.latency(env.time + sequence_length)

                state = next_state

                step_count += 1
                if step_count >= 1000:  # Terminate after 2000 steps
                    done = True
                    break  # Exit the loop when termination condition is met

            result_data.append({'Time': env.time, 'Throughput': state[-1][0], 'Latency': state[-1][4],
                                'CPU Usage': state[-1][2], 'Memory Usage': state[-1][3], 'num_vms': num_vms})

            if done:
                break  # Exit the outer loop when termination condition is met

    result_df = pd.DataFrame(result_data)
    result_df.to_csv(f'output_{eval_steps}.csv', index=False)
    print(f"Evaluation results saved to 'output_{eval_steps}.csv'.")


if __name__ == '__main__':
    train_steps_list = [2000, 5000]
    eval_steps = 2000
    max_episodes_per_step = 1000

    for train_steps in train_steps_list:
        print(f"Training for {train_steps} steps...")
        train_agent(train_steps, max_episodes_per_step)
        print("Training completed.\n")

    print("Training completed for all steps.\n")

    for train_steps in train_steps_list:
        print(f"Evaluating using model trained with {train_steps} steps...")
        evaluate_agent(train_steps)
        print(f"Evaluation completed for model trained with {train_steps} steps.\n")

    print("All evaluations completed.")

