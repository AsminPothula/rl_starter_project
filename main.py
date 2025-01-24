import numpy as np
import itertools
from sliding_puzzle_env import SlidingPuzzleEnv
from tqdm import tqdm
from tabulate import tabulate
from plot_training_results import plot_training_results


def main():
    puzzle_size = 2
    env = SlidingPuzzleEnv(grid_size=puzzle_size)
    all_states = list(itertools.permutations(np.arange(puzzle_size**2)))
    state_to_index = {state: index for index, state in enumerate(all_states)}

    Q_table = np.zeros((len(all_states), env.action_space.n))  # 24 states, 4 actions

    learning_rate = 0.8 
    exploration_prob = 0.2 
    discount_factor = 0.95 
    epochs = 100

    # ✅ Initialize lists to store training stats
    epochs_list = []
    steps_list = []
    rewards_list = []

    with tqdm(total=epochs, desc="Training Progress", bar_format="{l_bar}{bar} [ {n_fmt}/{total_fmt} epochs ]") as pbar:
        for epoch in range(epochs):
            state = env.reset() 
            done = False

            steps = 0
            total_reward = 0
            actions_taken = []
            exploration_count = 0
            exploitation_count = 0
            step_details = []  

            print(f"\nEpoch {epoch + 1}/{epochs} Running...\n")
            print(f"{'Step':<6}{'Action':<10}{'Reward':<10}")
            print("-" * 30)

            while not done:
                state_index = state_to_index[tuple(state)] 

                if np.random.rand() < exploration_prob:
                    action = np.random.randint(0, env.action_space.n)  # Explore
                    exploration_count += 1
                else:
                    action = np.argmax(Q_table[state_index])  # Exploit 
                    exploitation_count += 1

                next_state, reward, done, _ = env.step(action)
                next_state_index = state_to_index[tuple(next_state)]  # Get the index of the next state

                Q_table[state_index, action] += learning_rate * (
                    reward + discount_factor * np.max(Q_table[next_state_index]) - Q_table[state_index, action]
                )

                steps += 1
                total_reward += reward
                action_name = ['Up', 'Down', 'Left', 'Right'][action]
                actions_taken.append(action_name)
                step_details.append([steps, action_name, f"{reward:.2f}"])
                print(f"{steps:<6}{action_name:<10}{reward:<10.2f}")

                state = next_state

            table_data = [
                ["Total Steps", steps],
                ["Total Reward", f"{total_reward:.2f}"],
                ["Actions Taken", ", ".join(actions_taken)],
                ["Exploration Count", exploration_count],
                ["Exploitation Count", exploitation_count]
            ]

            print("\nEpoch Summary:")
            print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))
            print("-" * 80)

            # ✅ Store statistics per epoch
            epochs_list.append(epoch + 1)
            steps_list.append(steps)
            rewards_list.append(total_reward)

            pbar.update(1)

    print("Learned Q-table:")
    print(Q_table)
    plot_training_results(epochs_list, steps_list, rewards_list)
   
   #testing
    print("Training completed, testing the agent:")
    state = env.reset() 
    done = False
    while not done:
        env.render()
        state_index = state_to_index[tuple(state)]
        action = np.argmax(Q_table[state_index])
        state, reward, done, _ = env.step(action)
        print(f"Action: {['Up', 'Down', 'Left', 'Right'][action]} | Reward: {reward}")

    print("Congratulations! The agent solved the puzzle!")
    env.render()


if __name__ == "__main__":
    main()

