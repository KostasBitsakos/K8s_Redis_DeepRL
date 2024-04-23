import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000
NUM_EPISODES = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transition namedtuple to store experiences
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Neural Network for Q-Learning
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(5, 24)  # Input: 4 metrics + 1 VM count
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 3)  # Output: actions (decrease, maintain, increase VMs)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Memory
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Environment for VM Management
class VMEnvironment:
    def __init__(self):
        self.data = pd.read_csv('system_metrics.csv')
        self.num_vms = 6
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        self.num_vms = 6
        return self.get_state()

    def step(self, action):
        if action == 0 and self.num_vms > 1:
            self.num_vms -= 1
        elif action == 2:
            self.num_vms += 1

        self.current_step += 1
        done = self.current_step >= len(self.data)
        if done:
            return torch.zeros(1, 5).to(DEVICE), 0, done  # Default state, zero reward, done
        return self.get_state(), self.calculate_reward(), done

    def get_state(self):
        if self.current_step < len(self.data):
            metrics = self.data.iloc[self.current_step][['Throughput', 'Latency', 'CPU Usage', 'Memory Usage']].values
            return torch.from_numpy(np.append(metrics, [self.num_vms])).float().unsqueeze(0).to(DEVICE)
        else:
            return torch.zeros(1, 5).to(DEVICE)  # Return a default state if out of bounds

    def calculate_reward(self):
        if self.current_step < len(self.data):
            metrics = self.data.iloc[self.current_step]
            return (4.0 * metrics['Throughput'] - 1.0 * metrics['Latency'] - 1.0 * metrics['CPU Usage'] - 1.0 * metrics['Memory Usage'])
        return 0

def select_action(state, policy_net, steps_done):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(3)]], device=DEVICE, dtype=torch.long)

def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return  # Not enough samples to optimize

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action).unsqueeze(1)  # Ensuring it is [batch_size, 1]
    if action_batch.shape[1:] != (1,):
        action_batch = action_batch.squeeze(-1)  # Correcting if there's an extra unwanted dimension

    action_batch = action_batch.long()  # Ensure it's the correct type for indexing
    reward_batch = torch.cat(batch.reward)

    # Debugging print statements to confirm shapes before gather
    print(f"State batch shape: {state_batch.shape}")  # Expected: [batch_size, num_features]
    print(f"Action batch shape: {action_batch.shape}")  # Should be [batch_size, 1]
    print(f"Policy network output shape: {policy_net(state_batch).shape}")  # Expected: [batch_size, num_actions]
    print(f"Action batch type: {action_batch.dtype}")  # Expected: torch.long

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool, device=DEVICE)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    if non_final_next_states.size(0) > 0:
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()







def train_model():
    env = VMEnvironment()
    policy_net = DQN().to(DEVICE)
    target_net = DQN().to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)
    steps_done = 0

    episode_rewards = []  # Track rewards for plotting
    vm_counts = []  # Track VM counts for plotting
    metrics_data = []  # Track metrics for plotting

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0  # Reset total reward each episode
        episode_steps = 0  # Track steps per episode for plotting
        for _ in range(1000):  # Modify as necessary for episode length
            action = select_action(state, policy_net, steps_done)
            steps_done += 1
            next_state, reward, done = env.step(action.item())
            reward_tensor = torch.tensor([reward], device=DEVICE)  # Convert reward to tensor immediately
            total_reward += reward_tensor.item()  # Sum up rewards for the episode

            memory.push(state, action, next_state, reward_tensor)
            state = next_state

            optimize_model(memory, policy_net, target_net, optimizer)
            if done:
                break

            # Collect data for plotting
            if env.current_step < len(env.data):
                metrics = env.data.iloc[env.current_step]
                metrics_data.append(metrics[['Throughput', 'Latency', 'CPU Usage', 'Memory Usage']].tolist())
                vm_counts.append(env.num_vms)
                episode_steps += 1

        episode_rewards.append(total_reward)  # Append total reward of this episode

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode + 1}/{NUM_EPISODES}, Total Reward: {total_reward}")

    print("Training complete.")

    # Convert lists to numpy arrays for plotting
    metrics_data = np.array(metrics_data)
    vm_counts = np.array(vm_counts)

    # Plotting
    fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    metric_names = ['Throughput', 'Latency', 'CPU Usage', 'Memory Usage']
    colors = ['blue', 'green', 'red', 'purple']

    for i, ax in enumerate(axs):
        ax.plot(metrics_data[:, i], label=f'{metric_names[i]}', color=colors[i])
        ax.scatter(range(len(vm_counts)), [m[i] for m in metrics_data], color='red', label='VM Count on Metrics')
        ax.legend(loc='upper right')
        ax.set_ylabel(metric_names[i])
    axs[-1].set_xlabel('Time Step')
    plt.show()

train_model()
