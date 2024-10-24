from environments.base_env import BaseEnv
from environments.commonroad_env.experiment_setup import create_base_experiment_config, setup_base_experiment
from environments.commonroad_env.observers import create_representation_observer
from environments.commonroad_env.rewarders import create_rewarders
from environments.commonroad_env.termination_criteria import create_termination_criteria
from environments.commonroad_env.callbacks import CommonRoadWandbCallback
from commonroad_geometric.learning.reinforcement.rewarder.reward_aggregator.implementations import SumRewardAggregator
from omegaconf import DictConfig, OmegaConf

class CommonRoadEnv(BaseEnv):
    def __init__(self):
        self.experiment = None
        self.env = None

    def make_env(self, config, n_envs, seed, rl_mode=False):
        experiment_config = create_base_experiment_config(OmegaConf.to_container(config, resolve=True))
        experiment_config.termination_criteria = create_termination_criteria()
        if rl_mode:
            experiment_config.env_options.observer = create_representation_observer(config, config['device'])
            experiment_config.rewarder = SumRewardAggregator(create_rewarders())
            experiment_config.respawner_options['init_steering_angle'] = 0.0
            experiment_config.respawner_options['init_orientation_noise'] = 0.0
            experiment_config.respawner_options['init_position_noise'] = 0.0
            experiment_config.respawner_options['min_goal_distance_l2'] = 400.0
            experiment_config.respawner_options['route_length'] = 1
            experiment_config.respawner_options['min_vehicle_distance'] = 10.0
            experiment_config.respawner_options['future_timestep_count'] = 5
            experiment_config.respawner_options['init_speed'] = 'auto'
            experiment_config.control_space_options['lower_bound_acceleration'] = -10.0
            experiment_config.control_space_options['upper_bound_acceleration'] = 10.0

        self.experiment, self.env = setup_base_experiment(config, experiment_config)
        return self.env

    def get_observation_space(self):
        if self.env is None:
            raise ValueError("Environment has not been created yet. Call make_env() first.")
        return self.env.observation_space

    def get_action_space(self):
        if self.env is None:
            raise ValueError("Environment has not been created yet. Call make_env() first.")
        return self.env.action_space
    
    def custom_callbacks(self, config):
        return [CommonRoadWandbCallback(config)]