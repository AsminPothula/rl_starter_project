import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class SlidingPuzzleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=2, tile_paths=None):
        super(SlidingPuzzleEnv, self).__init__()

        self.grid_size = grid_size
        self.num_tiles = grid_size * grid_size
        self.empty_tile = self.num_tiles - 1  # Last tile is empty

        # Load image tiles and assign each number (0-3) to an image
        self.tile_paths = tile_paths
        self.tiles = {i: mpimg.imread(tile_paths[i]) for i in range(len(tile_paths))}

        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)

        # State: Flat array representing the puzzle
        self.observation_space = spaces.Box(
            low=0,
            high=self.num_tiles - 1,
            shape=(self.num_tiles,),
            dtype=np.int32
        )

        self.goal_state = list(range(self.num_tiles))
        self.state = None

    def reset(self):
        self.state = self._create_solvable_puzzle()
        return np.array(self.state)

    def step(self, action):
        grid = np.array(self.state).reshape(self.grid_size, self.grid_size)
        empty_pos = np.argwhere(grid == self.empty_tile)[0]
        move = None

        if action == 0 and empty_pos[0] > 0:  # Up
            move = (-1, 0)
        elif action == 1 and empty_pos[0] < self.grid_size - 1:  # Down
            move = (1, 0)
        elif action == 2 and empty_pos[1] > 0:  # Left
            move = (0, -1)
        elif action == 3 and empty_pos[1] < self.grid_size - 1:  # Right
            move = (0, 1)
        else:
            return np.array(self.state), -100, False, {}

        target_pos = empty_pos + move
        grid[empty_pos[0], empty_pos[1]], grid[target_pos[0], target_pos[1]] = \
            grid[target_pos[0], target_pos[1]], grid[empty_pos[0], empty_pos[1]]

        self.state = grid.flatten().tolist()
        reward = self._calculate_reward()
        done = self.state == self.goal_state

        return np.array(self.state), reward, done, {}

    def render(self, mode='human', save_path=None):
        fig, ax = plt.subplots(self.grid_size, self.grid_size, figsize=(4, 4))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
                if self.state[idx] != self.empty_tile:
                    ax[i, j].imshow(self.tiles[self.state[idx]])
                ax[i, j].axis("on")

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            print(f"Saved puzzle state to {save_path}")
        else:
            plt.show()
        plt.close()


    def _create_solvable_puzzle(self):
        state = self.goal_state.copy()
        while True:
            np.random.shuffle(state)
            if self._is_solvable(state) and state != self.goal_state:
                return state

    def _is_solvable(self, state):
        inversion_count = 0
        tiles = [tile for tile in state if tile != self.empty_tile]
        for i in range(len(tiles)):
            for j in range(i + 1, len(tiles)):
                if tiles[i] > tiles[j]:
                    inversion_count += 1

        if self.grid_size % 2 == 1:
            return inversion_count % 2 == 0
        else:
            empty_row = state.index(self.empty_tile) // self.grid_size
            return inversion_count % 2 != (empty_row % 2)

    def _calculate_reward(self):
        return sum([1 if self.state[i] == self.goal_state[i] else 0 for i in range(self.num_tiles)]) + \
               (100 if self.state == self.goal_state else 0)
