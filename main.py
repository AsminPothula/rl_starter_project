from sliding_puzzle_env import SlidingPuzzleEnv

#def action_to_string(action):
#    return {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}.get(action, 'Invalid')

def string_to_action(s):
    s = s.lower()
    return {'up': 0, 'down': 1, 'left': 2, 'right': 3}.get(s, -1)

def main():
    env = SlidingPuzzleEnv(grid_size=2)
    state = env.reset()
    done = False

    while not done:
        env.render()
        action_str = input("Enter your action (Up/Down/Left/Right): ")
        action = string_to_action(action_str)
        if action == -1:
            print("Invalid action. Please enter Up, Down, Left, or Right.")
            continue

        next_state, reward, done, _ = env.step(action)
        print(f"Reward: {reward}")
        state = next_state

        if done:
            print("Congratulations! You solved the puzzle!")
            env.render()
            break

if __name__ == "__main__":
    main()
