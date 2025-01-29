import numpy as np
import itertools
import os
import matplotlib.pyplot as plt
from sliding_puzzle_env import SlidingPuzzleEnv
from tqdm import tqdm
from tabulate import tabulate
from plot_training_results import plot_training_results

"""def save_puzzle_state(env, state, filename):
    #Saves a visualization of the puzzle state as an image.
    grid_size = env.grid_size
    fig, ax = plt.subplots(grid_size, grid_size, figsize=(4, 4))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if state[idx] != env.empty_tile:
                ax[i, j].imshow(env.tiles[state[idx]])
            ax[i, j].axis("off")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()"""

def main():
    # File to store all training statistics and messages
    output_file = "training_output2.txt"
    
    # Load tile images
    tile_folder = "puzzle_tiles"
    tile_paths = [os.path.join(tile_folder, f"tile_{i}.png") for i in range(4)]

    # Initialize the environment with image tiles
    puzzle_size = 2
    env = SlidingPuzzleEnv(grid_size=puzzle_size, tile_paths=tile_paths)

    # Generate all possible states
    all_states = list(itertools.permutations(np.arange(puzzle_size**2)))
    state_to_index = {state: index for index, state in enumerate(all_states)}

    # Initialize Q-table
    Q_table = np.zeros((len(all_states), env.action_space.n))

    # Define parameters
    learning_rate = 0.8
    exploration_prob = 0.2
    discount_factor = 0.95
    epochs = 1000  

    # Initialize lists for training statistics (for graphing)
    epochs_list = []
    steps_list = []
    rewards_list = []

    # Open file for logging
    with open(output_file, "w") as f:
        # Training
        with tqdm(total=epochs, desc="Training Progress", bar_format="{l_bar}{bar} [ {n_fmt}/{total_fmt} epochs ]") as pbar:
            for epoch in range(epochs):
                state = env.reset()
                done = False

                # Tracking statistics
                steps = 0
                total_reward = 0
                actions_taken = []
                exploration_count = 0
                exploitation_count = 0

                # Write epoch header to log file
                f.write(f"\nEpoch {epoch + 1}/{epochs} Running...\n")
                f.write(f"{'Step':<6}{'Action':<10}{'Reward':<10}\n")
                f.write("-" * 30 + "\n")

                while not done:
                    state_index = state_to_index[tuple(state)]

                    # Choose action with epsilon-greedy strategy
                    if np.random.rand() < exploration_prob:
                        action = np.random.randint(0, env.action_space.n)  # Explore
                        exploration_count += 1
                    else:
                        action = np.argmax(Q_table[state_index])  # Exploit
                        exploitation_count += 1

                    # Take the action in the environment
                    next_state, reward, done, _ = env.step(action)
                    next_state_index = state_to_index[tuple(next_state)]

                    # Update Q-table
                    Q_table[state_index, action] += learning_rate * (
                        reward + discount_factor * np.max(Q_table[next_state_index]) - Q_table[state_index, action]
                    )

                    # Track step details
                    steps += 1
                    total_reward += reward
                    action_name = ['Up', 'Down', 'Left', 'Right'][action]
                    actions_taken.append(action_name)

                    # Log step details to file
                    f.write(f"{steps:<6}{action_name:<10}{reward:<10.2f}\n")

                    # Move to next state
                    state = next_state

                # Store statistics per epoch (for graphing)
                epochs_list.append(epoch + 1)
                steps_list.append(steps)
                rewards_list.append(total_reward)

                # Print summary table after epoch completion
                table_data = [
                    ["Total Steps", steps],
                    ["Total Reward", f"{total_reward:.2f}"],
                    ["Actions Taken", ", ".join(actions_taken)],
                    ["Exploration Count", exploration_count],
                    ["Exploitation Count", exploitation_count]
                ]
                f.write("\nEpoch Summary:\n")
                f.write(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid") + "\n")
                f.write("-" * 80 + "\n")

                pbar.update(1)

        f.write("\nLearned Q-table:\n")
        f.write(str(Q_table) + "\n")

    # Call the graphing function
    plot_training_results(epochs_list, steps_list, rewards_list)

    # After training, the Q-table represents the learned Q-values
    print("Training completed, testing the agent:")

    # Test the trained agent
    state = env.reset()
    done = False

    # Save the start state image
    #save_puzzle_state(env, state, "start_state.png")
    env.render(save_path="start_state.png")

    while not done:
        env.render()  # Visualize in terminal

        state_index = state_to_index[tuple(state)]
        action = np.argmax(Q_table[state_index])
        state, reward, done, _ = env.step(action)

        # Log testing details to file
        with open(output_file, "a") as f:
            f.write(f"Action: {['Up', 'Down', 'Left', 'Right'][action]} | Reward: {reward}\n")

    with open(output_file, "a") as f:
        f.write("\nPuzzle solved!\n")

    # Save the solved state image
    #save_puzzle_state(env, state, "solved_state.png")
    env.render(save_path="solved_state.png")

    env.render()

if __name__ == "__main__":
    main()
