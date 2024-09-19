import gymnasium as gym
from gymnasium import spaces
import numpy as np
from environments.base_env import BaseEnv
import cv2

class ImageCartPoleEnv(gym.Wrapper, BaseEnv):
    def __init__(self):
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        super().__init__(env)
        
        # Define the new observation space (3 color channels, 64x64 image)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._get_obs(), info

    def step(self, action):
        # Handle both scalar and array-like actions
        if isinstance(action, (np.ndarray, list)):
            action = action[0]
        if isinstance(action, np.ndarray):
            action = action.item()
        
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        img = self.env.render()
        img = cv2.resize(img, (64, 64))
        img = img.transpose(2, 0, 1)
        # Ensure the image is always (3, 64, 64) and uint8
        img = np.asarray(img, dtype=np.uint8)
        if img.shape != (3, 64, 64):
            raise ValueError(f"Unexpected observation shape: {img.shape}")
        return img

    def make_env(self, cfg, n_envs, seed, **kwargs):
        def _init():
            env = ImageCartPoleEnv()
            env.reset(seed=seed)
            return env

        if n_envs == 1:
            return _init()
        else:
            return gym.vector.SyncVectorEnv([lambda: _init() for _ in range(n_envs)])

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space

class VectorizedImageCartPoleEnv(BaseEnv):
    def __init__(self):
        self.single_env = ImageCartPoleEnv()

    def make_env(self, cfg, n_envs, seed, **kwargs):
        return self.single_env.make_env(cfg, n_envs, seed, **kwargs)

    def get_observation_space(self):
        return self.single_env.get_observation_space()

    def get_action_space(self):
        return self.single_env.get_action_space()