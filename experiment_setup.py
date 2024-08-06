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
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions, TrafficFeatureComputerOptions
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

def load_config():
    """Load configuration from a YAML file."""
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_rl_experiment_config(config):
    """Create an RLExperimentConfig based on the provided configuration."""
    rewarder = SumRewardAggregator([])  # Add reward computers as needed

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

    rl_experiment_config = RLExperimentConfig(
        control_space_cls=SteeringAccelerationSpace,
        control_space_options=config['control_space_options'],
        ego_vehicle_simulation_options=EgoVehicleSimulationOptions(
            vehicle_model=VehicleModel.KS,
            vehicle_type=VehicleType.BMW_320i
        ),
        env_options={
            'disable_graph_extraction': True,
            'raise_exceptions': True,
            'observer': RenderObserver(
                renderer=TrafficSceneRenderer(
                    options=TrafficSceneRendererOptions(
                        camera=EgoVehicleCamera(
                            view_range=config["viewer_options"]["view_range"],
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
                            # RenderTrafficGraphPlugin(),
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
                            # RenderObstacleFlowPlugin(
                            #     obstacle_line_width=0.0
                            # )
                        ],
                        viewer_options=GLViewerOptions(
                            window_height=config["viewer_options"]["window_size"],
                            window_width=config["viewer_options"]["window_size"],
                        )
                    )
                )
            )
        },
        respawner_cls=RandomRespawner,
        respawner_options=config['respawner_options'],
        rewarder=rewarder,
        simulation_cls=ScenarioSimulation,
        simulation_options=config['simulation_options'],
        termination_criteria=termination_criteria,
        traffic_extraction_options=TrafficExtractorOptions(
            edge_drawer=config['traffic_extraction_options']['edge_drawer'],
            postprocessors=config['traffic_extraction_options']['postprocessors'],
            only_ego_inc_edges=config['traffic_extraction_options']['only_ego_inc_edges'],
            assign_multiple_lanelets=config['traffic_extraction_options']['assign_multiple_lanelets'],
            ego_map_radius=config['traffic_extraction_options']['ego_map_radius'],
            include_lanelet_vertices=config['traffic_extraction_options']['include_lanelet_vertices'],
            feature_computers=feature_computers
        )
    )

    return rl_experiment_config

def setup_experiment(config):
    """Set up the RL experiment using the provided configuration."""
    rl_experiment_config = create_rl_experiment_config(config)
    experiment = RLExperiment(config=rl_experiment_config)

    # Create the environment
    environment = experiment.make_env(
        scenario_dir=Path(config["scenario_dir"]),
        n_envs=1,
        seed=config["seed"]
    )
    return experiment, environment
