from abc import ABC, abstractmethod
import gymnasium as gym

class BaseEnv(ABC):
    @abstractmethod
    def make_env(self, config: dict, n_envs, seed, **kwargs):
        pass

    @abstractmethod
    def get_observation_space(self):
        pass

    @abstractmethod
    def get_action_space(self):
        pass

    def custom_callbacks(self, config: dict):
        return []