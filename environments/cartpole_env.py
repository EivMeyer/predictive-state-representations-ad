from environments.base_env import BaseEnv
import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

class CartPoleEnv(BaseEnv):
    def make_env(self, config, n_envs, seed, **kwargs):
        def make_env_fn(rank):
            def _init():
                if config['cartpole']['render']:
                    env = gym.make("CartPole-v1", render_mode="human")
                else:
                    env = gym.make("CartPole-v1")
                env = TransformObservation(env, lambda x: x.astype(np.float32))
                env.reset(seed=seed + rank)
                return env
            return _init
            
        if n_envs == 1:
            return DummyVecEnv([make_env_fn(0)])
        return SubprocVecEnv([make_env_fn(i) for i in range(n_envs)])

    def get_observation_space(self):
        env = gym.make("CartPole-v1")
        return env.observation_space

    def get_action_space(self):
        env = gym.make("CartPole-v1") 
        return env.action_space