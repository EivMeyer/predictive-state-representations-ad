import hydra
from omegaconf import DictConfig, OmegaConf
from experiment_setup import setup_experiment
import time
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps

def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        wrapper.execution_times.append(end_time - start_time)
        return result
    wrapper.execution_times = []
    return wrapper

def instrument_env(env):
    env.step = time_function(env.step)
    env.reset = time_function(env.reset)
    
    if hasattr(env, 'env'):
        inner_env = env.env
        inner_env.step = time_function(inner_env.step)
        inner_env.reset = time_function(inner_env.reset)
    
    return env

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    experiment, env = setup_experiment(cfg_dict)
    
    env = instrument_env(env)
    
    num_episodes = 10
    steps_per_episode = 100
    all_step_times = []
    reset_times = []
    
    print(f"Running {num_episodes} episodes with {steps_per_episode} steps each...")
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}")
        
        start_time = time.time()
        env.reset()
        reset_time = time.time() - start_time
        reset_times.append(reset_time)
        
        episode_step_times = []
        for _ in range(steps_per_episode):
            actions = [env.action_space.sample() for _ in range(env.num_envs)]
            env.step(actions)
            episode_step_times.append(env.step.execution_times[-1])
        
        all_step_times.extend(episode_step_times)
    
    # Plotting
    plt.figure(figsize=(15, 10))
    
    # Plot step times across all episodes
    plt.subplot(2, 1, 1)
    plt.plot(all_step_times)
    plt.title('Step Execution Times Across Episodes')
    plt.xlabel('Step (across all episodes)')
    plt.ylabel('Time (seconds)')
    
    # Plot reset times
    plt.subplot(2, 1, 2)
    plt.plot(reset_times, 'ro-')
    plt.title('Reset Times for Each Episode')
    plt.xlabel('Episode')
    plt.ylabel('Time (seconds)')
    
    plt.tight_layout()
    plt.savefig('inter_episode_slowdown.png')
    print("Plot saved as inter_episode_slowdown.png")
    
    # Calculate and print statistics
    print("\nExecution Time Statistics:")
    print(f"Step times: Mean = {np.mean(all_step_times):.6f}s, Std = {np.std(all_step_times):.6f}s")
    print(f"Reset times: Mean = {np.mean(reset_times):.6f}s, Std = {np.std(reset_times):.6f}s")
    
    # Calculate and print the average step time for each episode
    step_times_per_episode = np.array_split(all_step_times, num_episodes)
    for i, episode_times in enumerate(step_times_per_episode):
        print(f"Episode {i+1} average step time: {np.mean(episode_times):.6f}s")

if __name__ == "__main__":
    main()