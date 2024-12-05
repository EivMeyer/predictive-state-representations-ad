import gymnasium as gym
from environments.base_env import BaseEnv
from environments.image_cartpole_env import ImageCartPoleEnv
def simulate_image_cartpole():
    # Create the environment with human-friendly rendering
    # env = ImageCartPoleEnv()
    env = gym.make("CartPole-v1", render_mode="human")
    # Run for 5 episodes
    for episode in range(5):
        obs, info = env.reset()
        total_reward = 0
        done = False
        step = 0
        while not done:
            # Render the environment (this will display in a window)
            env.render()
            # Take a random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1
        print(f"Episode {episode + 1} finished with total reward: {total_reward} and {step} steps")
    env.close()
if __name__ == "__main__":
    simulate_image_cartpole()