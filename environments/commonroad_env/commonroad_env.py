from environments.base_env import BaseEnv
from environments.commonroad_env.experiment_setup import create_base_experiment_config, setup_base_experiment
from environments.commonroad_env.observers import create_representation_observer
from environments.commonroad_env.rewarders import create_rewarders
from environments.commonroad_env.termination_criteria import create_termination_criteria
from commonroad_geometric.learning.reinforcement.rewarder.reward_aggregator.implementations import SumRewardAggregator

class CommonRoadEnv(BaseEnv):
    def __init__(self):
        self.experiment = None
        self.env = None

    def make_env(self, config, n_envs, seed, rl_mode=False):
        experiment_config = create_base_experiment_config(config['environments']['commonroad'])
        if rl_mode:
            experiment_config.env_options.observer = create_representation_observer(config, config['device'])
            experiment_config.rewarder = SumRewardAggregator(create_rewarders())
            experiment_config.termination_criteria = create_termination_criteria()

        self.experiment, self.env = setup_base_experiment(config)
        return self.env

    def get_observation_space(self):
        if self.env is None:
            raise ValueError("Environment has not been created yet. Call make_env() first.")
        return self.env.observation_space

    def get_action_space(self):
        if self.env is None:
            raise ValueError("Environment has not been created yet. Call make_env() first.")
        return self.env.action_space