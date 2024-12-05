from environments.base_env import BaseEnv
import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

class MountainCarEnv(BaseEnv):
    def make_env(self, config, n_envs, seed, **kwargs):
        def make_env_fn(rank):
            def _init():
                if config['mountain_car']['render']:
                    env = gym.make("MountainCar-v0", render_mode="human")
                else:
                    env = gym.make("MountainCar-v0")
                # Convert observations to float32
                env = TransformObservation(env, lambda x: x.astype(np.float32))
                env.reset(seed=seed + rank)
                return env
            return _init
            
        if n_envs == 1:
            return DummyVecEnv([make_env_fn(0)])
        return SubprocVecEnv([make_env_fn(i) for i in range(n_envs)])

    def get_observation_space(self):
        env = gym.make("MountainCar-v0")
        return env.observation_space

    def get_action_space(self):
        env = gym.make("MountainCar-v0") 
        return env.action_space
