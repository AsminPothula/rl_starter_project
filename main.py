import numpy as np
import itertools
from sliding_puzzle_env import SlidingPuzzleEnv

def main():
    # Initialize the environment
    puzzle_size = 3
    env = SlidingPuzzleEnv(grid_size=puzzle_size)

    # Generate all possible states (permutations of the tiles) 
    #all_states = list(itertools.permutations([1, 2, 3, 4, 5, 6, 7, 8, 0])) # generates all possible states & converts into list
    all_states = list(itertools.permutations(np.arange(puzzle_size**2)))
    # this basically makes a dictionary and assign index numbers to states
    # follows { key: value for item in iterable }
    state_to_index = {state: index for index, state in enumerate(all_states)}

    # Initialize Q-table
    Q_table = np.zeros((len(all_states), env.action_space.n))  # 24 states, 4 actions

    # Define parameters - ?
    learning_rate = 0.8 #fast updates (aggressive learning)
    exploration_prob = 0.2 #probability of taking a random action
    discount_factor = 0.95 #how much future rewards matter. here, prioritising long term results 
    epochs = 1000

    # Q-learning algorithm
    for epoch in range(epochs):
        state = env.reset()  # Reset the environment to a random state
        done = False

        while not done:
            # Get the index of the current state 
            # the state list is converted to a tuple, and its corresponding index is searched for (in s_to_in) and stored (in s_i)
            state_index = state_to_index[tuple(state)] #the dictionary is made and the states and indices are mapped, and the indices are stored in state_index

            # Choose action with epsilon-greedy strategy
            if np.random.rand() < exploration_prob:
                action = np.random.randint(0, env.action_space.n)  # Explore
            else:
                action = np.argmax(Q_table[state_index])  # Exploit 

            # Take the action in the environment
            next_state, reward, done, _ = env.step(action)
            next_state_index = state_to_index[tuple(next_state)]  # Get the index of the next state

            # Update Q-value using the Q-learning formula 
            # learning rate * difference between the new expected reward and the current Q-value
            Q_table[state_index, action] += learning_rate * (
                reward + discount_factor * np.max(Q_table[next_state_index]) - Q_table[state_index, action]
            )

            # Move to the next state
            state = next_state

        print(f"Epoch {epoch + 1}/{epochs} completed.")
    print("Learned Q-table:")
    print(Q_table)
    # q table is set up now
    # have to give a start state, the agent will solve it using the q values 

    # After training, the Q-table represents the learned Q-values
    print("Training completed, testing the agent:")

    # Test the trained agent
    state = env.reset() #here it is picking a random start state, i can instead give a specific start state
    done = False
    while not done:
        env.render() #visualise

        # Get the index of the current state 
        state_index = state_to_index[tuple(state)]

        # Select the best action 
        action = np.argmax(Q_table[state_index])

        # Take the action
        state, reward, done, _ = env.step(action)
        print(f"Action: {['Up', 'Down', 'Left', 'Right'][action]} | Reward: {reward}")

    print("Congratulations! The agent solved the puzzle!")
    env.render()


if __name__ == "__main__":
    main()
