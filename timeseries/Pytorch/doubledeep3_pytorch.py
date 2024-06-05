import numpy as np
import pandas as pd
import numpy as np
# Define the environment
import numpy as np
from environment import Environment
from ddqn import DDQNAgent


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
        from tqdm import tqdm
        while not done:
            for _ in tqdm(range(max_episodes_per_step)):
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

