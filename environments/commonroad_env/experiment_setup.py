# experiment_setup.py

from omegaconf import OmegaConf
import yaml
from pathlib import Path
from commonroad_geometric.learning.reinforcement.experiment import RLExperiment, RLExperimentConfig, RLEnvironmentOptions, RLEnvironmentParams
from commonroad_geometric.learning.reinforcement.rewarder.reward_aggregator.implementations import SumRewardAggregator
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions, TrafficFeatureComputerOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TrafficExtractorFactory
from commonroad_geometric.simulation.ego_simulation.control_space.implementations import PIDControlSpace, SteeringAccelerationSpace
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.implementations import *
from commonroad_geometric.learning.reinforcement.termination_criteria.implementations import *
from commonroad_geometric.common.io_extensions.scenario import LaneletAssignmentStrategy
from commonroad_geometric.dataset.scenario.preprocessing.wrappers.chain_preprocessors import chain_preprocessors
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulationOptions
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulation, ScenarioSimulationOptions
from commonroad_geometric.simulation.ego_simulation.respawning.implementations import RandomRespawner, RandomRespawnerOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from commonroad.common.solution import VehicleType, VehicleModel
from functools import partial
from environments.commonroad_env.rewarders import create_rewarders
from environments.commonroad_env.termination_criteria import create_termination_criteria
from environments.commonroad_env.observers import create_representation_observer, create_render_observer, create_renderer_options, create_representation_model


def create_scenario_preprocessors():
    scenario_preprocessors = [
        # VehicleFilterPreprocessor(),
        # RemoveIslandsPreprocessor()
        # SegmentLaneletsPreprocessor(100.0),
        # ComputeVehicleVelocitiesPreprocessor(),
        # (DepopulateScenarioPreprocessor(1), 1),
    ]
    return scenario_preprocessors

def create_base_experiment_config(config):
    """Create an RLExperimentConfig based on the provided configuration."""
    rewarder = SumRewardAggregator([])  # Add reward computers as needed

    termination_criteria = [TimeoutCriterion(500)]

    feature_computers = TrafficFeatureComputerOptions(
        v=[],
        v2v=[],
        l=[],
        l2l=[],
        v2l=[],
        l2v=[]
    )

    renderer_options_render = create_renderer_options(
        view_range=150,
        window_size=800
    )
    
    experiment_config = RLExperimentConfig(
        control_space_cls=SteeringAccelerationSpace,
        control_space_options=config['control_space'],
        ego_vehicle_simulation_options=EgoVehicleSimulationOptions(
            vehicle_model=VehicleModel.KS,
            vehicle_type=VehicleType.BMW_320i
        ),
        env_options=RLEnvironmentOptions(
            disable_graph_extraction=True,
            raise_exceptions=True,
            renderer_options=renderer_options_render,
            observer=create_render_observer(config['viewer']),
            preprocessor=chain_preprocessors(*create_scenario_preprocessors()),
        ),
        respawner_cls=RandomRespawner,
        respawner_options=config['respawner'],
        rewarder=rewarder,
        simulation_cls=ScenarioSimulation,
        simulation_options=config['simulation'],
        termination_criteria=termination_criteria,
        traffic_extraction_factory=TrafficExtractorFactory(options=TrafficExtractorOptions(
            edge_drawer=config['traffic_extraction']['edge_drawer'],
            postprocessors=config['traffic_extraction']['postprocessors'],
            only_ego_inc_edges=config['traffic_extraction']['only_ego_inc_edges'],
            assign_multiple_lanelets=config['traffic_extraction']['assign_multiple_lanelets'],
            ego_map_radius=config['traffic_extraction']['ego_map_radius'],
            include_lanelet_vertices=config['traffic_extraction']['include_lanelet_vertices'],
            feature_computers=feature_computers
        ))
    )
    
    experiment_config.simulation_options['lanelet_assignment_order'] = LaneletAssignmentStrategy.ONLY_CENTER 

    return experiment_config

def setup_base_experiment(config):
    """Set up the RL experiment using the provided configuration."""
 
    experiment_config = create_base_experiment_config(OmegaConf.to_container(config['envronments']['commonroad'], resolve=True))
    experiment = RLExperiment(config=experiment_config)

    # Create the environment
    environment = experiment.make_env(
        scenario_dir=Path(config['envronments']['commonroad']["scenario_dir"]),
        n_envs=config["dataset"]["num_workers"],
        seed=config["seed"]
    )
    return experiment, environment

def setup_rl_experiment(cfg):
    """
    Configures the downstream RL experiment by modifying the base experiment.
    """
    representation_observer_constructor = partial(create_representation_observer, cfg=cfg)

    experiment_config = create_base_experiment_config(OmegaConf.to_container(cfg, resolve=True))
    experiment_config.env_options.observer = representation_observer_constructor
    experiment_config.respawner_options['init_steering_angle'] = 0.0
    experiment_config.respawner_options['init_orientation_noise'] = 0.0
    experiment_config.respawner_options['init_position_noise'] = 0.0
    experiment_config.respawner_options['min_goal_distance_l2'] = 400.0
    experiment_config.respawner_options['route_length'] = 1
    experiment_config.respawner_options['min_vehicle_distance'] = 20.0
    experiment_config.respawner_options['init_speed'] = 'auto'
    experiment_config.control_space_options['lower_bound_acceleration'] = -10.0
    experiment_config.control_space_options['upper_bound_acceleration'] = 10.0
    experiment_config.rewarder = SumRewardAggregator(create_rewarders())
    experiment_config.termination_criteria = create_termination_criteria()

    experiment = RLExperiment(config=experiment_config)

    return experiment