# experiment_setup.py

import yaml
from pathlib import Path
from commonroad_geometric.learning.reinforcement.experiment import RLExperiment, RLExperimentConfig
from commonroad_geometric.learning.reinforcement.rewarder.reward_aggregator.implementations import SumRewardAggregator
from commonroad_geometric.learning.reinforcement.termination_criteria.implementations import (
    CollisionCriterion,
    OffroadCriterion,
    ReachedGoalCriterion,
    TimeoutCriterion
)
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.implementations import *
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.types import RewardLossMetric
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions, TrafficFeatureComputerOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TrafficExtractorFactory
from commonroad_geometric.simulation.ego_simulation.control_space.implementations import PIDControlSpace, SteeringAccelerationSpace
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulationOptions
from commonroad_geometric.learning.reinforcement.observer.implementations import RenderObserver
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer, TrafficSceneRendererOptions
from commonroad_geometric.rendering.plugins.cameras.ego_vehicle_camera import EgoVehicleCamera
from commonroad_geometric.rendering.plugins.implementations import (
    RenderLaneletNetworkPlugin,
    RenderPlanningProblemSetPlugin,
    RenderTrafficGraphPlugin,
    RenderEgoVehiclePlugin
)
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_plugin import RenderObstaclePlugin
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_flow_plugin import RenderObstacleFlowPlugin
from commonroad_geometric.rendering.viewer.pyglet.gl_viewer_2d import GLViewerOptions
from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulation, ScenarioSimulationOptions
from commonroad_geometric.simulation.ego_simulation.respawning.implementations import RandomRespawner, RandomRespawnerOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from commonroad.common.solution import VehicleType, VehicleModel

def create_rewarders():
    rewarders = [
        # AccelerationPenaltyRewardComputer(
        #     weight=0.0,
        #     loss_type=RewardLossMetric.L2
        # ),
        CollisionPenaltyRewardComputer(
            penalty=-1.5,
        ),
        # FrictionViolationPenaltyRewardComputer(penalty=-0.01),
        TrajectoryProgressionRewardComputer(
            weight=0.1,
            delta_threshold=0.08
        ),
        ConstantRewardComputer(reward=-0.001),
        #
        ReachedGoalRewardComputer(reward=3.5),
        OvershotGoalRewardComputer(reward=0.0),
        # SteeringAnglePenaltyRewardComputer(weight=0.0005, loss_type=RewardLossMetric.L1),
        StillStandingPenaltyRewardComputer(penalty=-0.05, velocity_threshold=2.0),
        # TimeToCollisionPenaltyRewardComputer(weight=0.1), # requires incoming edges
        OffroadPenaltyRewardComputer(penalty=-3.5),
        VelocityPenaltyRewardComputer(
            reference_velocity=28.0,
            weight=0.002,
            loss_type=RewardLossMetric.L2,
            only_upper=True
        ),

        LateralErrorPenaltyRewardComputer(weight=0.0001, loss_type=RewardLossMetric.L1),
        YawratePenaltyRewardComputer(weight=0.01),
        # HeadingErrorPenaltyRewardComputer(
        #     weight=0.01,
        #     loss_type=RewardLossMetric.L2,
        #     wrong_direction_penalty=-0.01
        # )
    ]

    return rewarders

def create_renderer_options(view_range, window_size):
    renderer_options = TrafficSceneRendererOptions(
        camera=EgoVehicleCamera(
            view_range=view_range,
            camera_rotation_speed=None
        ),
        plugins=[
            RenderLaneletNetworkPlugin(
                lanelet_linewidth=0.0,
                fill_color=Color("grey")
            ),
            RenderPlanningProblemSetPlugin(
                render_trajectory=False,
                render_start_waypoints=False,
                render_goal_waypoints=True,
                render_look_ahead_point=False
            ),
            RenderEgoVehiclePlugin(
                render_trail=False,
                ego_vehicle_linewidth=0.0,
                ego_vehicle_color_collision=None,
                ego_vehicle_fill_color=Color((0.1, 0.8, 0.1, 1.0))
            ),
            RenderObstaclePlugin(
                from_graph=False,
                obstacle_fill_color=Color("red"),
                obstacle_color=Color("red"),
                obstacle_line_width=0.0
            ),
        ],
        viewer_options=GLViewerOptions(
            window_height=window_size,
            window_width=window_size,
        )
    )

    return renderer_options

def create_render_observer(config):
    renderer_options = create_renderer_options(
        view_range=config["view_range"],
        window_size=config["window_size"]
    )
    return RenderObserver(
        renderer_options=renderer_options
    )

def create_rl_experiment_config(config):
    """Create an RLExperimentConfig based on the provided configuration."""
    rewarder = SumRewardAggregator(create_rewarders())  # Add reward computers as needed

    termination_criteria = [
        # CollisionCriterion(),
        # OffroadCriterion(),
        # ReachedGoalCriterion()
        TimeoutCriterion(max_timesteps=500)
    ]

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
        window_size=1200
    )
    
    rl_experiment_config = RLExperimentConfig(
        control_space_cls=SteeringAccelerationSpace,
        control_space_options=config['control_space'],
        ego_vehicle_simulation_options=EgoVehicleSimulationOptions(
            vehicle_model=VehicleModel.KS,
            vehicle_type=VehicleType.BMW_320i
        ),
        env_options={
            'disable_graph_extraction': True,
            'raise_exceptions': True,
            'renderer_options': renderer_options_render,
            'observer': create_render_observer(config['viewer']),
        },
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

    return rl_experiment_config

def setup_experiment(config):
    """Set up the RL experiment using the provided configuration."""
    rl_experiment_config = create_rl_experiment_config(config)
    experiment = RLExperiment(config=rl_experiment_config)

    # Create the environment
    environment = experiment.make_env(
        scenario_dir=Path(config["scenario_dir"]),
        n_envs=config["dataset"]["num_workers"],
        seed=config["seed"]
    )
    return experiment, environment
