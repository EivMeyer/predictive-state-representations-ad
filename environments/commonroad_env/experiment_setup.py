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
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.implementations import *
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulationOptions
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulation, ScenarioSimulationOptions
from commonroad_geometric.simulation.ego_simulation.respawning.implementations import *
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from commonroad.common.solution import VehicleType, VehicleModel
from environments.commonroad_env.observers import create_representation_observer
from functools import partial
from environments.commonroad_env.control_space import TrackVehicleControlSpace
from environments.commonroad_env.rewarders import create_rewarders
from environments.commonroad_env.termination_criteria import create_termination_criteria
from utils.rl_utils import create_representation_model
from environments.commonroad_env.observers import create_render_observer, create_renderer_options


def create_base_experiment_config(config):
    """Create an RLExperimentConfig based on the provided configuration."""
    rewarder = SumRewardAggregator([])  # Add reward computers as needed

    commonroad_config = config['commonroad']

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

    control_space_cls = PIDControlSpace if commonroad_config['pid_control'] else SteeringAccelerationSpace
    control_space_options = commonroad_config['control_space']

    preprocessors = [
        # MergeLaneletsPreprocessor()
        # VehicleFilterPreprocessor(),
        # RemoveIslandsPreprocessor()
        # SegmentLaneletsPreprocessor(100.0),
        # ComputeVehicleVelocitiesPreprocessor(),
        # (DepopulateScenarioPreprocessor(1), 1),
    ]
    if commonroad_config['spawn_rate'] < 1.0:
        preprocessors.append(DepopulateScenarioPreprocessor(commonroad_config["spawn_rate"]))
    
    experiment_config = RLExperimentConfig(
        control_space_cls=control_space_cls,
        control_space_options=control_space_options,
        ego_vehicle_simulation_options=EgoVehicleSimulationOptions(
            vehicle_model=VehicleModel.KS,
            vehicle_type=VehicleType.BMW_320i
        ),
        env_options=RLEnvironmentOptions(
            disable_graph_extraction=True,
            raise_exceptions=True,
            renderer_options=renderer_options_render,
            num_respawns_per_scenario=commonroad_config['num_respawns_per_scenario'],
            observer=create_render_observer(config['viewer']),
            preprocessor=chain_preprocessors(*preprocessors) if preprocessors else None,
        ),
        respawner_cls=RandomRespawner,
        respawner_options=commonroad_config['respawner'],
        rewarder=rewarder,
        simulation_cls=ScenarioSimulation,
        simulation_options=commonroad_config['simulation'],
        termination_criteria=termination_criteria,
        traffic_extraction_factory=TrafficExtractorFactory(options=TrafficExtractorOptions(
            edge_drawer=commonroad_config['traffic_extraction']['edge_drawer'],
            postprocessors=commonroad_config['traffic_extraction']['postprocessors'],
            only_ego_inc_edges=commonroad_config['traffic_extraction']['only_ego_inc_edges'],
            assign_multiple_lanelets=commonroad_config['traffic_extraction']['assign_multiple_lanelets'],
            ego_map_radius=commonroad_config['traffic_extraction']['ego_map_radius'],
            include_lanelet_vertices=commonroad_config['traffic_extraction']['include_lanelet_vertices'],
            feature_computers=feature_computers
        ))
    )
    
    experiment_config.simulation_options['lanelet_assignment_order'] = LaneletAssignmentStrategy.ONLY_CENTER 

    return experiment_config

def setup_base_experiment(config, experiment_config):
    """Set up the RL experiment using the provided configuration."""

    if isinstance(config, OmegaConf):
        config = OmegaConf.to_container(config, resolve=True)
 
    experiment = RLExperiment(config=experiment_config)

    # Create the environment
    environment = experiment.make_env(
        scenario_dir=Path(config['commonroad']["scenario_dir"]),
        n_envs=config["dataset"]["num_workers"],
        seed=config["seed"]
    )
    return experiment, environment

def setup_rl_experiment(cfg):
    """
    Configures the downstream RL experiment by modifying the base experiment.
    """

    experiment_config = create_base_experiment_config(OmegaConf.to_container(cfg, resolve=True))

    if not cfg.rl_training.online_srl:
        representation_observer_constructor = partial(create_representation_observer, cfg=cfg)
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
    experiment_config.termination_criteria = create_termination_criteria(terminate_on_collision=not cfg['dataset']['collect_from_trajectories'], terminate_on_timeout=True)

    experiment = RLExperiment(config=experiment_config)

    return experiment