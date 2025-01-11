import numpy as np
from sliding_puzzle_env import SlidingPuzzleEnv


def main():
    env = SlidingPuzzleEnv(grid_size=2)
    #state = env.reset()

    # Initialize Q-table with zeros
    #Q_table = np.zeros((env.num_tiles, env.action_space.n))
    Q_table = np.zeros((24, env.action_space.n))

    # Define parameters
    learning_rate = 0.8
    discount_factor = 0.95
    exploration_prob = 0.2
    epochs = 100

    
    # Q-learning algorithm
    for epoch in range(epochs):
        state = env.reset() # Reset the environment to a random state / start from a random state
        # env.render() # visualise
        done = False

        while not done: # basically means when current state != goal state
            # visualise the env/ current state 
            # env.render()

            if np.random.rand() < exploration_prob:
                action = np.random.randint(0, env.action_space.n)  # Explore
            else:
                action = np.argmax(Q_table[state])  # Exploit

            next_state, reward, done, _ = env.step(action)
                
            Q_table[state, action] += learning_rate * (reward + discount_factor * np.max(Q_table[next_state]) - Q_table[state, action])
            
            state = next_state
        # when the goal state for this random start state is reached, one epoch is completed
    print("Learned Q-table:")
    print(Q_table)

    
    #Q table is set up comepletely now. We now give the agent a particular start state, and it solves it using the Q-table
    # by picking the action at every state with the highest q value

    state = env.reset() #here we are starting with a random state, we can also specify a particular start state
    done = False # i.e., goal state not reached
    print("Start State:")
    env.render() #visualise start state
    
    while not done: # basically means when current state != goal state
        # env.render()
       
        # Select the best/highest q-value action from the trained Q-table
        action = np.argmax(Q_table[state])  # Exploit the learned policy

        # Take the action
        next_state, reward, done, _ = env.step(action)
        # print(f"Action: {['Up', 'Down', 'Left', 'Right'][action]} | Reward: {reward}")

        state = next_state

    print("Congratulations! The agent solved the puzzle!")
    print("Final Achieved/Solved State:")
    env.render()
    

if __name__ == "__main__":
    main()
