import gym
from gym import spaces
import numpy as np
import os
import cv2

class SlidingPuzzleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=2, image_dir="output_images", goal_state=None):
        super(SlidingPuzzleEnv, self).__init__()

        self.grid_size = grid_size
        self.num_tiles = grid_size * grid_size
        self.image_dir = image_dir  
        self.empty_tile = 0  

        # ✅ Use the dynamically assigned goal state from main.py
        self.goal_state = goal_state if goal_state else list(range(1, self.num_tiles)) + [0]

        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)

        # State: Flat array representing the puzzle
        self.observation_space = spaces.Box(
            low=0,
            high=self.num_tiles - 1,
            shape=(self.num_tiles,),
            dtype=np.int32
        )

        self.state = None  # Initialize state

    def reset(self):
        """Randomly shuffles the puzzle and returns the initial state."""
        self.state = self._create_solvable_puzzle()
        return np.array(self.state)

    def step(self, action):
        """Moves the empty tile based on the action taken."""
        grid = np.array(self.state).reshape(self.grid_size, self.grid_size)
        empty_pos = np.argwhere(grid == self.empty_tile)[0]  # Get empty tile position

        print(f"🔹 Current State: {self.state}")  # ✅ Print current state
        print(f"🔹 Empty Tile Position: {empty_pos}")  # ✅ Debugging
        print(f"🔹 Action Attempted: {['Up', 'Down', 'Left', 'Right'][action]}")

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
            print("🚨 Invalid Move - Empty tile cannot move in that direction.")
            return np.array(self.state), -100, False, {}

        target_pos = empty_pos + move  # Get new empty tile position

        # Swap tiles
        grid[empty_pos[0], empty_pos[1]], grid[target_pos[0], target_pos[1]] = \
            grid[target_pos[0], target_pos[1]], grid[empty_pos[0], empty_pos[1]]

        self.state = grid.flatten().tolist()

        reward = self._calculate_reward()
        done = self._is_goal_state()

        print(f"✅ New State After Move: {self.state}")
        print(f"🏁 Goal Reached? {done} | Reward: {reward}\n")

        return np.array(self.state), reward, done, {}


    def _is_goal_state(self):
        """Checks if the puzzle is solved."""
        return self.state == self.goal_state  # ✅ Compare dynamically assigned goal state

    def _create_solvable_puzzle(self):
        """Generates a solvable puzzle with one empty tile."""
        while True:
            state = np.random.permutation(self.num_tiles).tolist()
            if self._is_solvable(state) and state != self.goal_state:
                print(f"✅ Generated Solvable Puzzle: {state}")  # Debugging
                return state

    def _is_solvable(self, state):
        """Checks if a given puzzle configuration is solvable."""
        inversions = sum(1 for i in range(len(state)) for j in range(i+1, len(state))
                         if state[i] > state[j] and state[i] != self.empty_tile and state[j] != self.empty_tile)
        return inversions % 2 == 0

    def _calculate_reward(self):
        """Computes reward based on proximity to goal state."""
        reward = sum(1 for i, tile in enumerate(self.state) if tile == self.goal_state[i])
        if self._is_goal_state():
            reward += 100
        return reward

    def render(self):
        """Visualizes the puzzle in a simple text format."""
        grid = np.array(self.state).reshape(self.grid_size, self.grid_size)
        print("\nCurrent Puzzle State:")
        for row in grid:
            print(' '.join([str(tile) if tile != self.empty_tile else ' ' for tile in row]))
        print("-" * (self.grid_size * 4))
