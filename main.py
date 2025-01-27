import numpy as np
import itertools
import os
import matplotlib.pyplot as plt
from sliding_puzzle_env import SlidingPuzzleEnv
from tqdm import tqdm
from tabulate import tabulate

def save_puzzle_state(env, state, filename):
    """ Saves a visualization of the puzzle state as an image. """
    grid_size = env.grid_size
    fig, ax = plt.subplots(grid_size, grid_size, figsize=(4, 4))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if state[idx] != env.empty_tile:
                ax[i, j].imshow(env.tiles[state[idx]])
            ax[i, j].axis("off")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved puzzle state to {filename}")

def main():
    # Load tile images
    tile_folder = "puzzle_tiles"
    tile_paths = [os.path.join(tile_folder, f"tile_{i}.png") for i in range(4)]

    # Ensure tiles exist
    for path in tile_paths:
        if not os.path.exists(path):
            print(f"ERROR: Tile {path} not found!")
            return

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

    # Training
    with tqdm(total=epochs, desc="Training Progress") as pbar:
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs} running...")
            state = env.reset()
            done = False

            while not done:
                state_index = state_to_index[tuple(state)]

                if np.random.rand() < exploration_prob:
                    action = np.random.randint(0, env.action_space.n)
                else:
                    action = np.argmax(Q_table[state_index])

                next_state, reward, done, _ = env.step(action)
                next_state_index = state_to_index[tuple(next_state)]

                Q_table[state_index, action] += learning_rate * (
                    reward + discount_factor * np.max(Q_table[next_state_index]) - Q_table[state_index, action]
                )

                state = next_state

            pbar.update(1)

    print("\nTraining completed. Testing the agent:")

    # Start Testing
    state = env.reset()
    done = False

    # Save the start state image
    save_puzzle_state(env, state, "start_state.png")

    while not done:
        env.render()
        state_index = state_to_index[tuple(state)]
        action = np.argmax(Q_table[state_index])
        state, reward, done, _ = env.step(action)
        print(f"Action: {['Up', 'Down', 'Left', 'Right'][action]} | Reward: {reward}")

    print("\nPuzzle solved!")

    # Save the solved state image
    save_puzzle_state(env, state, "solved_state.png")

    env.render()

if __name__ == "__main__":
    main()
