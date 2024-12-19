import gym
from gym import spaces
import numpy as np

class SlidingPuzzleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=3):
        super(SlidingPuzzleEnv, self).__init__()

        self.grid_size = grid_size
        self.num_tiles = grid_size * grid_size
        self.empty_tile = 0  # Represent the empty space with 0

        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)

        # State: Flat array representing the puzzle
        self.observation_space = spaces.Box(
            low=0,
            high=self.num_tiles - 1,
            shape=(self.num_tiles,),
            dtype=np.int32
        )

        self.goal_state = np.arange(1, self.num_tiles).tolist() + [self.empty_tile]
        self.state = None

    def reset(self):
        self.state = self._create_solvable_puzzle()
        return np.array(self.state)

    def step(self, action):
        grid = np.array(self.state).reshape(self.grid_size, self.grid_size) #converts the flat list into a numpy 2D arry grid
        empty_pos = np.argwhere(grid == self.empty_tile)[0] #finds the position of the empty tile (0,0) (0,1) (1,0) (1,1)
        move = None
        #figures out if it is a valid move or not (if so, tells the empty tile what move to do to perform that action)
        if action == 0 and empty_pos[0] > 0:  # Up
            move = (-1, 0) #stay in same column, move up (back)
        elif action == 1 and empty_pos[0] < self.grid_size - 1:  # Down
            move = (1, 0)
        elif action == 2 and empty_pos[1] > 0:  # Left
            move = (0, -1)
        elif action == 3 and empty_pos[1] < self.grid_size - 1:  # Right
            move = (0, 1)
        else:
            # Invalid move
            reward = -100 #more intense negative reward so this path isnt explored in the future 
            done = False
            return np.array(self.state), reward, done, {} #return and end it here and ask for next move

        target_pos = empty_pos + move # target position/location to move to
        # Swap tiles
        grid[empty_pos[0], empty_pos[1]], grid[target_pos[0], target_pos[1]] = \
            grid[target_pos[0], target_pos[1]], grid[empty_pos[0], empty_pos[1]]

        self.state = grid.flatten().tolist()

        # Calculate reward
        reward = self._calculate_reward()

        # Check if the puzzle is solved
        done = self.state == self.goal_state

        return np.array(self.state), reward, done, {}

    def render(self, mode='human'): #visualizes the puzzle 
        grid = np.array(self.state).reshape(self.grid_size, self.grid_size) #converst into numpy grid
        print("-" * (self.grid_size * 4)) #prints top bar
        for row in grid: #prints each row
            print(' '.join([str(tile) if tile != self.empty_tile else ' ' for tile in row])) 
        print("-" * (self.grid_size * 4)) #prints bottom bar  

    def close(self):
        pass

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
        # For odd grid sizes
        if self.grid_size % 2 == 1:
            return inversion_count % 2 == 0
        else:
            # For even grid sizes, consider the position of the empty tile
            empty_row = state.index(self.empty_tile) // self.grid_size
            if empty_row % 2 == 0:
                return inversion_count % 2 == 1
            else:
                return inversion_count % 2 == 0

    def _calculate_reward(self):
        reward = 0
        for current_tile, goal_tile in zip(self.state, self.goal_state):
            if current_tile == goal_tile:
                reward += 1
        if reward == self.num_tiles:
            reward += 100
        return reward
