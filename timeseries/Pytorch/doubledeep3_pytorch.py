import numpy as np
import pandas as pd
# Define the environment
from environment import Environment
from ddqn import DDQNAgent
import matplotlib.pyplot as plt



def train_agent(train_steps, max_episodes_per_step=1000):
    env = Environment()
    state_size = 6
    action_size = 3
    sequence_length = 10
    batch_size = 16
    agent = DDQNAgent(state_size, action_size, sequence_length)

    rewards = []
    episode_loads = []
    episode_read_percentages = []
    episode_mem_usages = []
    episode_cpu_usages = []
    episode_latencies=[]
    episode_num_vms = []

    for _ in range(train_steps):
        state = np.zeros((sequence_length, state_size))
        total_reward = 0
        no_improvement_counter = 0
        best_reward = -np.inf


        episode_load = []
        episode_read_percentage = []
        episode_mem_usage = []
        episode_cpu_usage = []
        episode_latency=[]
        episode_num_vm = []
        
        done = False  # Initialize done outside the loop
        step_count = 0  # Initialize step_count
        from tqdm import tqdm
        while not done:
            for episode_idx in tqdm(range(max_episodes_per_step)):
                action = agent.act(state, verbose=0)

                new_state, reward, done = env.step(action)
                total_reward += reward
                # print("here")
                # print(new_state)
                # print("Shape of next_state:", new_state.shape)

                # next_state = np.zeros((sequence_length, state_size))
                # next_state[:-1] = state[1:]
                # next_state[-1][0] = env.load(env.time + sequence_length)
                # next_state[-1][1] = env.read_percentage(env.time + sequence_length)
                # next_state[-1][2] = env.cpu_usage(env.time + sequence_length)
                # next_state[-1][3] = env.memory_usage(env.time + sequence_length)
                # next_state[-1][4] = env.latency(env.time + sequence_length)
                
                # print(next_state)
                
                
                next_state = np.zeros((sequence_length, state_size))
                next_state[:-1] = state[1:]
                next_state[-1][0] = new_state[0]
                next_state[-1][1] = new_state[1]
                next_state[-1][2] = new_state[2]
                next_state[-1][3] = new_state[3]
                next_state[-1][4] = new_state[4]
                next_state[-1][5] = new_state[5]

                #print(next_state)
                agent.remember(state, action, reward, next_state, done)

                episode_load.append(next_state[-1][0])
                episode_read_percentage.append(next_state[-1][1])
                episode_mem_usage.append(next_state[-1][2])
                episode_cpu_usage.append(next_state[-1][3])
                episode_latency.append(next_state[-1][4])

                episode_num_vm.append(next_state[-1][5])

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

                if episode_idx % 2 == 0:
                    agent.plot_loss_history()

                if episode_idx % 10 == 0:
                    evaluate_agent(agent, 100)

            rewards.append(total_reward)
            episode_loads.append(episode_load)
            episode_read_percentages.append(episode_read_percentage)
            episode_mem_usages.append(episode_mem_usage)
            episode_cpu_usages.append(episode_cpu_usage)
            episode_latencies.append(episode_latency)
            episode_num_vms.append(episode_num_vm)
            if done:
                break  # Exit the outer loop when termination condition is met


    with open(f"rewards_{train_steps}.txt", "w") as f:
        for reward in rewards:
            f.write(str(reward) + "\n")

    agent.save_model(f'model_{train_steps}.pth')
    # print(f"Model for {train_steps} training steps saved to 'model_{train_steps}.h5'.")
    # print("Training completed.\n")
    return episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms


def evaluate_agent(agent, eval_steps):
    env = Environment()
    state_size = 6  # Adjusted state size to include latency
    action_size = 3
    sequence_length = 10
    batch_size = 16
    # agent = DDQNAgent(state_size, action_size, sequence_length)

    # Load the trained model weights
    # agent.online_model.load_weights(f'model_{train_steps}.weights.h5')

    total_rewards = []
    episode_loads = []
    episode_read_percentages = []
    episode_mem_usages = []
    episode_cpu_usages = []
    episode_latencies = []  # Added to store latency values
    episode_num_vms = []

    for _ in range(1):
        state = np.zeros((sequence_length, state_size))
        total_reward = 0

        episode_load = []
        episode_read_percentage = []
        episode_mem_usage = []
        episode_cpu_usage = []
        episode_latency = []  # Added to store latency values
        episode_num_vm = []

        done = False
        idx = 0
        while not done:
            idx += 1
            action = agent.act(state, verbose=0)

            new_state, reward, done = env.step(action)
            total_reward += reward

            next_state = np.zeros((sequence_length, state_size))
            next_state[:-1] = state[1:]
            next_state[-1][0] = new_state[0]
            next_state[-1][1] = new_state[1]
            next_state[-1][2] = new_state[2]
            next_state[-1][3] = new_state[3]
            next_state[-1][4] = new_state[4]
            next_state[-1][5] = new_state[5]

            #print(next_state)
            agent.remember(state, action, reward, next_state, done)

            episode_load.append(next_state[-1][0])
            episode_read_percentage.append(next_state[-1][1])
            episode_mem_usage.append(next_state[-1][2])
            episode_cpu_usage.append(next_state[-1][3])
            episode_latency.append(next_state[-1][4])

            episode_num_vm.append(next_state[-1][5])

            state = next_state

            if idx > eval_steps:
                break

        total_rewards.append(total_reward)
        episode_loads.append(episode_load)
        episode_read_percentages.append(episode_read_percentage)
        episode_mem_usages.append(episode_mem_usage)
        episode_cpu_usages.append(episode_cpu_usage)
        episode_latencies.append(episode_latency)  # Append latency episodes
        episode_num_vms.append(episode_num_vm)

    avg_reward = np.mean(total_rewards)
    print(f"Avg. reward for model trained with {train_steps} steps: {avg_reward}")
    
    return episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms


def plot_metrics(episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms):
    # Plotting metrics
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    for episode_load in episode_loads:
        plt.plot(range(len(episode_load)), episode_load, alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Load')
    plt.title('Load Over Time')

    plt.subplot(3, 2, 2)
    for episode_read_percentage in episode_read_percentages:
        plt.plot(range(len(episode_read_percentage)), episode_read_percentage, alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Read Percentage')
    plt.title('Read Percentage Over Time')

    plt.subplot(3, 2, 3)
    for episode_mem_usage in episode_mem_usages:
        plt.plot(range(len(episode_mem_usage)), episode_mem_usage, alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Memory Usage')
    plt.title('Memory Usage Over Time')

    plt.subplot(3, 2, 4)
    for episode_cpu_usage in episode_cpu_usages:
        plt.plot(range(len(episode_cpu_usage)), episode_cpu_usage, alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('CPU Usage')
    plt.title('CPU Usage Over Time')

    plt.subplot(3, 2, 5)
    for episode_latency in episode_latencies:
        plt.plot(range(len(episode_latency)), episode_latency, alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Latency')
    plt.title('Latency Over Time')

    plt.subplot(3, 2, 6)
    for episode_num_vm in episode_num_vms:
        plt.plot(range(len(episode_num_vm)), episode_num_vm, alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Number of VMs')
    plt.title('Number of VMs Over Time')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    train_steps_list = [2000, 5000]
    eval_steps = 2000
    max_episodes_per_step = 1000

    for train_steps in train_steps_list:
        print(f"Training for {train_steps} steps...")
        episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms = train_agent(train_steps, max_episodes_per_step)
        print("Training completed.\n")

    print("Training completed for all steps.\n")

    for train_steps in train_steps_list:
        print(f"Evaluating using model trained with {train_steps} steps...")
        episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms = evaluate_agent(agent, eval_steps)
        plot_metrics(episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms)

        print(f"Evaluation completed for model trained with {train_steps} steps.\n")

    print("All evaluations completed.")



