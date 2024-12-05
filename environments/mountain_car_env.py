from environments.base_env import BaseEnv
import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gymnasium.envs.classic_control


# Custom random initial state MountainCar environment
class RandomMountainCarGymEnv(gymnasium.envs.classic_control.mountain_car.MountainCarEnv):
    def reset(self, seed=None, options=None):
        # Set seed if provided
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Random initial state: position and velocity
        self.state = np.array([
            self.np_random.uniform(low=-1.2, high=0.6),  # Random position
            self.np_random.uniform(low=-0.04, high=0.04)  # Random velocity
        ])
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32), {}

# Base environment wrapper for multi-env setup
class MountainCarEnv(BaseEnv):
    def make_env(self, config, n_envs, seed, **kwargs):
        def make_env_fn(rank):
            def _init():
                if config['mountain_car']['render']:
                    env = RandomMountainCarGymEnv(render_mode="human")
                else:
                    env = RandomMountainCarGymEnv()
                # Convert observations to float32
                env = TransformObservation(env, lambda x: x.astype(np.float32))
                env.reset(seed=seed + rank)
                return env
            return _init
        
        if n_envs == 1:
            return DummyVecEnv([make_env_fn(0)])
        return SubprocVecEnv([make_env_fn(i) for i in range(n_envs)])

    def get_observation_space(self):
        # Use the custom environment to get the observation space
        env = RandomMountainCarGymEnv()
        return env.observation_space

    def get_action_space(self):
        # Use the custom environment to get the action space
        env = RandomMountainCarGymEnv()
        return env.action_space
