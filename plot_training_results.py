import os
import matplotlib.pyplot as plt

def plot_training_results(epochs_list, steps_list, rewards_list, output_dir="training_results"):
    """Generate and save training graphs."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Graph 1: Steps per Epoch
    plt.figure(figsize=(10, 4))
    plt.plot(epochs_list, steps_list, linestyle='-', color='b')
    plt.xlabel("Epochs")
    plt.ylabel("Steps Taken")
    plt.title("Steps per Epoch")
    plt.grid()
   # plt.ylim(bottom=-100, top=100)  # change these values to view the extremes (-200,200)
    plt.savefig(os.path.join(output_dir, "steps_per_epoch.png"))
    plt.close()
    
    # Graph 2: Total Reward per Epoch
    plt.figure(figsize=(10, 4))
    plt.plot(epochs_list, rewards_list, linestyle='-', color='g')
    plt.xlabel("Epochs")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Epoch")
    plt.grid()
    #plt.ylim(bottom=-250, top=250)  # change these values to view the extremes (-500,500)
    plt.savefig(os.path.join(output_dir, "reward_per_epoch.png"))
    plt.close()
    
    print(f"Graphs saved in {output_dir} directory.")
