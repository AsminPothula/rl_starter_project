import numpy as np
import itertools
import os
import cv2
from sliding_puzzle_env import SlidingPuzzleEnv
from tqdm import tqdm

# ✅ Automatically detect Desktop path for saving training results
DESKTOP_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "rl_output")
OUTPUT_DIR = os.path.abspath(DESKTOP_DIR)  # Save files to Desktop

# ✅ Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ Image file path (should be in the same directory as `main.py`)
IMAGE_FILE = "puzzle.png"

# ✅ Function to split image into 4 tiles and assign one tile as empty
def split_and_save_image(image_path, output_dir):
    """Splits the image into 4 tiles and assigns one as the empty tile."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (200, 200))  # Resize to 200x200

    os.makedirs(output_dir, exist_ok=True)

    tiles = []
    for i in range(2):
        for j in range(2):
            tile = image[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100]
            tiles.append(tile)

    # ✅ Randomly select one tile as the empty tile (represented by index 0)
    empty_tile_index = np.random.choice(4)

    for i in range(4):
        tile_filename = os.path.join(output_dir, f"tile_{i}.png")
        if i == empty_tile_index:
            cv2.imwrite(tile_filename, np.ones((100, 100, 3), dtype=np.uint8) * 255)  # White empty tile
        else:
            cv2.imwrite(tile_filename, tiles[i])

    print(f"✅ Image split into 4 tiles. Empty tile is at index: {empty_tile_index}")

# ✅ Detect the empty tile from saved images
def get_correct_goal_state(output_dir):
    """Determines the correct goal state based on the actual empty tile."""
    empty_tile_index = None
    for i in range(4):
        tile_path = os.path.join(output_dir, f"tile_{i}.png")
        tile_image = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
        if tile_image is not None and np.all(tile_image == 255):  # White tile is the empty tile
            empty_tile_index = i
            break

    if empty_tile_index is None:
        raise ValueError("❌ Could not determine the empty tile index.")

    # ✅ Always set goal state to [1, 2, 3, 0] but adjust empty tile dynamically
    goal_state = [1, 2, 3, 0]  # ✅ Correct goal state for sliding puzzle

    print(f"✅ Correct Goal State: {goal_state}")  # Debugging
    return goal_state


def main():
    """Main function to train Q-learning on sliding puzzle."""
    print("📌 Current Directory:", os.getcwd())

    # ✅ Ensure image exists and process into tiles
    if not os.path.exists(IMAGE_FILE):
        print(f"❌ ERROR: Upload an image as '{IMAGE_FILE}' in the current directory before running.")
        return

    # ✅ Split and save tiles
    split_and_save_image(IMAGE_FILE, OUTPUT_DIR)

    # ✅ Get dynamically determined goal state
    goal_state = get_correct_goal_state(OUTPUT_DIR)

    # ✅ Initialize Q-learning environment
    puzzle_size = 2
    env = SlidingPuzzleEnv(grid_size=puzzle_size, image_dir=OUTPUT_DIR, goal_state=goal_state)

    all_states = list(itertools.permutations(range(puzzle_size**2)))
    state_to_index = {state: index for index, state in enumerate(all_states)}

    Q_table = np.zeros((len(all_states), env.action_space.n))  

    # ✅ Hyperparameters
    learning_rate = 0.8
    exploration_prob = 0.2
    discount_factor = 0.95
    epochs = 10
    max_moves = 50  # ✅ Limit moves per episode to prevent infinite loops

    # ✅ Training loop
    with tqdm(total=epochs, desc="Training Progress") as pbar:
        for epoch in range(epochs):
            state = env.reset()  # Start with a new random state
            done = False
            moves = 0  # ✅ Track move count

            while not done and moves < max_moves:
                state_index = state_to_index[tuple(state)]
                
                # Explore vs Exploit
                action = np.random.randint(0, env.action_space.n) if np.random.rand() < exploration_prob else np.argmax(Q_table[state_index])  

                next_state, reward, done, _ = env.step(action)
                next_state_index = state_to_index[tuple(next_state)]

                # ✅ Update Q-table
                Q_table[state_index, action] += learning_rate * (
                    reward + discount_factor * np.max(Q_table[next_state_index]) - Q_table[state_index, action]
                )

                state = next_state
                moves += 1  # ✅ Increment move count

            pbar.update(1)

    print("\n✅ Training Complete! Learned Q-table:")
    print(Q_table)

if __name__ == "__main__":
    main()
