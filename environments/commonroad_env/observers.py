import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from commonroad_geometric.learning.reinforcement.experiment import RLExperiment, RLExperimentConfig
from utils.dataset_utils import EnvironmentDataset, get_data_dimensions
from utils.file_utils import find_model_path
from utils.training_utils import print_parameter_summary

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
import numpy as np
import wandb
import gymnasium
from functools import partial
import numpy as np
import torch
from typing import Optional, Tuple, Set, Dict
from collections import deque
import matplotlib.pyplot as plt
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.observer.base_observer import BaseObserver, T_Observation
from commonroad_geometric.learning.reinforcement.observer.implementations.render_observer import RenderObserver
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from commonroad_geometric.learning.reinforcement.termination_criteria.implementations import *
from commonroad_geometric.learning.reinforcement.termination_criteria.base_termination_criterion import BaseTerminationCriterion
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.implementations import *
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.types import RewardLossMetric
from commonroad_geometric.learning.reinforcement.rewarder.reward_aggregator.implementations import SumRewardAggregator
from models import get_model_class
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import numpy as np
from matplotlib import font_manager

# Global configuration
SAVE_SEPARATE_PLOTS = False  # Set to True for separate plots, False for combined plot
SAVE_PLOTS = False

class RepresentationObserver(BaseObserver):
    def __init__(
        self, 
        dataset: EnvironmentDataset,
        representation_model, 
        device, 
        render_observer: RenderObserver, 
        sequence_length: int, 
        include_ego_state: bool = True,
        debug: bool = False,
        debug_freq: int = 1
    ):
        super().__init__()
        self.dataset = dataset
        self.representation_model = representation_model
        self.device = device
        self.render_observer = render_observer
        self.sequence_length = sequence_length
        self.include_ego_state = include_ego_state
        self.debug = debug
        self.debug_freq = debug_freq
        self.representation_model.eval()
        self.obs_buffer = deque(maxlen=sequence_length)
        self.ego_state_buffer = deque(maxlen=sequence_length)
        self.is_first_observation = True
        
        if self.debug:
            self.setup_debug_plot()
        self.call_count = 0

    def setup(self, dummy_data: Optional[CommonRoadData] = None) -> gymnasium.Space:
        render_space = self.render_observer.setup(dummy_data)

        dummy_obs = np.zeros((self.sequence_length, *render_space.shape), dtype=np.float32)
        dummy_ego_state = np.zeros((self.sequence_length, 4), dtype=np.float32)  # Assuming 4 ego state variables

        with torch.no_grad():
            dummy_obs_tensor = torch.from_numpy(dummy_obs).float().unsqueeze(0).permute(0, 1, 4, 2, 3).to(self.device)
            dummy_ego_state_tensor = torch.from_numpy(dummy_ego_state).float().unsqueeze(0).to(self.device)
            dummy_batch = {
                'observations': dummy_obs_tensor,
                'ego_states': dummy_ego_state_tensor
            }
            dummy_rep = self.representation_model.encode(dummy_batch)
            if isinstance(dummy_rep, tuple):
                dummy_rep = dummy_rep[0]

        rep_shape = dummy_rep.cpu().numpy().shape[1:]

        obs_shape = rep_shape if not self.include_ego_state else (rep_shape[0] + dummy_ego_state.shape[-1],)

        return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

    def observe(
        self,
        data: CommonRoadData,
        ego_vehicle_simulation: EgoVehicleSimulation
    ) -> T_Observation:
        render_obs_raw = self.render_observer.observe(data, ego_vehicle_simulation)
        render_obs = self.dataset.preprocess_image(render_obs_raw)
        if render_obs.ndim == 4:
            render_obs = render_obs[0, ...]  # Remove the first dimension (batch dimension)

        ego_state = np.array([
            ego_vehicle_simulation.ego_vehicle.state.velocity,
            0.0,
            ego_vehicle_simulation.ego_vehicle.state.__dict__.get('steering_angle', 0.0),
            ego_vehicle_simulation.ego_vehicle.state.__dict__.get('yaw_rate', 0.0)
        ])

        self.obs_buffer.append(render_obs)
        self.ego_state_buffer.append(ego_state)

        if self.is_first_observation:
            while len(self.obs_buffer) < self.sequence_length:
                self.obs_buffer.appendleft(render_obs)
                self.ego_state_buffer.appendleft(ego_state)
            self.is_first_observation = False

        obs_sequence = np.array(self.obs_buffer)
        ego_state_sequence = np.array(self.ego_state_buffer)

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_sequence).float().unsqueeze(0).permute(0, 1, 4, 2, 3).to(self.device)
            ego_state_tensor = torch.from_numpy(ego_state_sequence).float().unsqueeze(0).to(self.device)
            batch = {
                'observations': obs_tensor,
                'ego_states': ego_state_tensor
            }
            rep = self.representation_model.encode(batch)
            if isinstance(rep, tuple):
                rep = rep[0]
            
        representation = rep.cpu().numpy().squeeze()

        if self.debug and self.call_count % self.debug_freq == 0:
            with torch.no_grad():
                decoding_arr = self.representation_model.decode_image(batch, rep)
                if isinstance(decoding_arr, (tuple, list)):
                    decoding = decoding_arr[0]
                    hazard_preds = decoding_arr[1]
                    done_preds = decoding_arr[2]
                else:
                    decoding = decoding_arr
                    hazard_preds = None
                    done_preds = None
                if decoding.ndim == 5: # If the model returns a sequence of predictions
                    decoding = decoding[0]
                    hazard_preds = hazard_preds[0] if hazard_preds is not None else None
                    done_preds = done_preds[0] if done_preds is not None else None

            predictions = decoding.permute(0, 2, 3, 1).cpu().detach().numpy()
            hazard_preds = hazard_preds.cpu().detach().numpy() if hazard_preds is not None else None
            done_preds = done_preds.cpu().detach().numpy() if done_preds is not None else None

            self.update_debug_plot(render_obs, predictions, hazard_preds, done_preds, representation)

        if self.include_ego_state:
            representation = np.concatenate([representation, ego_state])

        self.call_count += 1

        return representation

    def reset(self, ego_vehicle_simulation: EgoVehicleSimulation) -> None:
        self.obs_buffer.clear()
        self.ego_state_buffer.clear()
        self.is_first_observation = True
        if self.debug:
            print("RepresentationObserver reset: Cleared observation and ego state buffers.")

    def get_debug_figure(self):
        """Get the current debug visualization figure."""
        if not self.debug:
            return None
        if SAVE_SEPARATE_PLOTS:
            return None  # We don't support separate plots for recording
        return self.fig

    def setup_debug_plot(self):
        # Set up LaTeX-like plotting style
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.usetex": True,
            "pgf.rcfonts": False,
        })

        # Increase the default font sizes
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
        })
        
        if SAVE_SEPARATE_PLOTS:
            self.setup_separate_plots()
        else:
            self.setup_combined_plot()

        plt.ion()
        plt.show()

    def setup_separate_plots(self):
        # Create three separate figures
        self.fig_obs = plt.figure(figsize=(5, 5))
        self.fig_pred = plt.figure(figsize=(10, 10))
        self.fig_rep = plt.figure(figsize=(5, 5))

        # Current observation
        self.ax_obs = self.fig_obs.add_subplot(111)
        self.im_obs = self.ax_obs.imshow(np.zeros((64, 64, 3)))
        self.ax_obs.axis('off')

        # Predictions grid
        self.ax_pred = self.fig_pred.add_subplot(111)
        self.ax_pred.axis('off')
        self.im_preds = []

        for i in range(3):
            for j in range(3):
                ax = self.ax_pred.inset_axes([j/3 + 0.02, (2-i)/3 + 0.08, 0.29, 0.29])
                im = ax.imshow(np.zeros((64, 64, 3)), cmap='viridis')
                self.im_preds.append(im)
                ax.axis('off')
                rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
                ax.add_patch(rect)
                step = i * 3 + j
                label = '$t$' if step == 0 else f'$t+{step}$'
                ax.set_title(label, fontsize=17, pad=8)

        # Latent representation
        self.ax_rep = self.fig_rep.add_subplot(111)
        self.im_rep = self.ax_rep.imshow(np.zeros((1, 1)), cmap='viridis', aspect='equal')
        cbar = self.fig_rep.colorbar(self.im_rep, ax=self.ax_rep)
        cbar.set_label('Value', fontsize=12)

    def setup_combined_plot(self):
        self.fig = plt.figure(figsize=(15, 5))
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 2, 1], wspace=0.3)

        # Current observation
        self.ax_obs = self.fig.add_subplot(gs[0, 0])
        self.im_obs = self.ax_obs.imshow(np.zeros((64, 64, 3)))
        self.ax_obs.axis('off')

        # Predictions grid
        self.ax_pred = self.fig.add_subplot(gs[0, 1])
        self.ax_pred.axis('off')
        self.im_preds = []

        for i in range(3):
            for j in range(3):
                ax = self.ax_pred.inset_axes([j/3 + 0.02, (2-i)/3 + 0.08, 0.29, 0.29])  # Adjust the vertical space by changing (2-i)/3 + 0.08 - specifically, the offset value, which is the vertical offset that moves the plots up or down
                im = ax.imshow(np.zeros((64, 64, 3)), cmap='viridis')
                self.im_preds.append(im)
                ax.axis('off')
                rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
                ax.add_patch(rect)
                step = i * 3 + j
                label = '$t$' if step == 0 else f'$t+{step}$'
                ax.set_title(label, fontsize=14, pad=-20, loc='center')  # Negative pad moves the title below the plot

        # Latent representation
        self.ax_rep = self.fig.add_subplot(gs[0, 2])
        self.im_rep = self.ax_rep.imshow(np.zeros((1, 1)), cmap='viridis', aspect='equal')
        cbar = self.fig.colorbar(self.im_rep, ax=self.ax_rep)
        cbar.set_label('Value', fontsize=12)

    def update_debug_plot(self, current_obs, predictions, hazard_preds, done_preds, representation):
        if SAVE_SEPARATE_PLOTS:
            self.update_separate_plots(current_obs, predictions, hazard_preds, done_preds, representation)
        else:
            self.update_combined_plot(current_obs, predictions, hazard_preds, done_preds, representation)

    def update_separate_plots(self, current_obs, predictions, hazard_preds, done_preds, representation):
        # Update current observation
        self.im_obs.set_data(current_obs)
        self.fig_obs.canvas.draw()
        if SAVE_PLOTS:
            self.fig_obs.savefig(f'./debug_plots/step_{self.call_count}_observation.pdf', dpi=100, bbox_inches='tight')

        # Update predictions
        for i, im in enumerate(self.im_preds):
            if i < len(predictions):
                pred = predictions[i]
                im.set_data(pred)
            else:
                im.set_data(np.zeros_like(current_obs))
            im.set_extent([0, current_obs.shape[1], current_obs.shape[0], 0])
        self.fig_pred.canvas.draw()
        if SAVE_PLOTS:
            self.fig_pred.savefig(f'./debug_plots/step_{self.call_count}_predictions.pdf', dpi=100, bbox_inches='tight')

        # Update latent representation
        rep_dim = int(np.sqrt(representation.shape[0]))
        rep_reshaped = representation.reshape(rep_dim, rep_dim)
        self.im_rep.set_data(rep_reshaped)
        self.im_rep.autoscale()
        self.fig_rep.canvas.draw()
        if SAVE_PLOTS:
            self.fig_rep.savefig(f'./debug_plots/step_{self.call_count}_representation.pdf', dpi=100, bbox_inches='tight')

    def update_combined_plot(self, current_obs, predictions, hazard_preds, done_preds, representation):
        # Update current observation
        self.im_obs.set_data(current_obs)

        # Update predictions
        for i, im in enumerate(self.im_preds):
            if i < len(predictions):
                pred = predictions[i]
                im.set_data(pred)
            else:
                im.set_data(np.zeros_like(current_obs))
            im.set_extent([0, current_obs.shape[1], current_obs.shape[0], 0])
            
            done_prob = done_preds[i] if done_preds is not None else 0
            done_prob_str = f"{done_prob:.2f}"
            im_title = f"t+{i} ({done_prob_str})"
            im.axes.set_title(im_title, fontsize=14, pad=-20, loc='center')

        # Update latent representation
        rep_dim = int(np.sqrt(representation.shape[0]))
        rep_reshaped = representation.reshape(rep_dim, rep_dim)
        self.im_rep.set_data(rep_reshaped)
        self.im_rep.autoscale()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if SAVE_PLOTS:
            self.fig.savefig(f'./debug_plots/step_{self.call_count}_combined.pdf', dpi=100, bbox_inches='tight')

def create_renderer_options(view_range: int, window_size: int, vehicle_expansion_factor: float):
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
            # RenderPlanningProblemSetPlugin(
            #     render_trajectory=True,
            #     render_start_waypoints=True,
            #     render_goal_waypoints=True,
            #     render_look_ahead_point=False
            # ),
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
                obstacle_line_width=0.0,
                vehicle_expansion=vehicle_expansion_factor
            ),
        ],
        viewer_options=GLViewerOptions(
            window_height=window_size,
            window_width=window_size,
        )
    )

    return renderer_options

def create_render_observer(view_config: dict, commonroad_config: dict):
    renderer_options = create_renderer_options(
        view_range=view_config["view_range"],
        window_size=view_config["window_size"],
        vehicle_expansion_factor=commonroad_config["vehicle_expansion_factor"]
    )
    return RenderObserver(
        renderer_options=renderer_options
    )

def create_representation_observer(cfg, device):
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = device
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        device = torch.device("cpu")
    cfg['device'] = str(device)

    from utils.rl_utils import create_representation_model
    representation_model = create_representation_model(cfg, device)
    print_parameter_summary(representation_model)
    render_observer = create_render_observer(view_config=cfg['viewer'], commonroad_config=cfg['commonroad'])
    dataset = EnvironmentDataset(cfg)
    representation_observer = RepresentationObserver(dataset, representation_model, device, debug=cfg['debug_mode'], render_observer=render_observer, sequence_length=cfg['dataset']['t_obs'])
    return representation_observer

def create_frame_stacking_observer(cfg):
    base_observer = create_render_observer(view_config=cfg['viewer'], commonroad_config=cfg['commonroad'])
    frame_stacking_observer = FrameStackingObserver(base_observer, stack_size=cfg['dataset']['t_obs'])
    return frame_stacking_observer


class FrameStackingObserver(BaseObserver):
    def __init__(self, base_observer: BaseObserver, stack_size: int):
        """
        Chains observations together over a specified number of timesteps.

        Args:
            base_observer (BaseObserver): The observer providing the base observations.
            stack_size (int): The number of observations to stack.
        """
        super().__init__()
        self.base_observer = base_observer
        self.stack_size = stack_size
        self.obs_buffer = deque(maxlen=stack_size)
        self.is_first_observation = True

    def setup(self, dummy_data: Optional[CommonRoadData] = None) -> gymnasium.Space:
        """
        Initialize the observation space based on the base observer.

        Args:
            dummy_data (Optional[CommonRoadData]): Dummy data for initialization.

        Returns:
            gymnasium.Space: The observation space of stacked observations.
        """
        base_space = self.base_observer.setup(dummy_data)
        stacked_shape = (self.stack_size, *base_space.shape)
        return gymnasium.spaces.Box(
            low=np.tile(base_space.low[np.newaxis, ...], (self.stack_size, 1, 1, 1)),
            high=np.tile(base_space.high[np.newaxis, ...], (self.stack_size, 1, 1, 1)),
            shape=stacked_shape,
            dtype=base_space.dtype
        )

    def observe(self, data: CommonRoadData, ego_vehicle_simulation: EgoVehicleSimulation) -> T_Observation:
        """
        Stack observations over the last `stack_size` timesteps.

        Args:
            data (CommonRoadData): The simulation data.
            ego_vehicle_simulation (EgoVehicleSimulation): Ego vehicle simulation instance.

        Returns:
            T_Observation: The stacked observation tensor.
        """
        observation = self.base_observer.observe(data, ego_vehicle_simulation)
        self.obs_buffer.append(observation)

        if self.is_first_observation:
            while len(self.obs_buffer) < self.stack_size:
                self.obs_buffer.appendleft(observation)
            self.is_first_observation = False

        stacked_observation = np.stack(self.obs_buffer, axis=0)
        return stacked_observation

    def reset(self, ego_vehicle_simulation: EgoVehicleSimulation) -> None:
        """
        Reset the buffer and propagate reset to the base observer.

        Args:
            ego_vehicle_simulation (EgoVehicleSimulation): Ego vehicle simulation instance.
        """
        self.obs_buffer.clear()
        self.is_first_observation = True
        self.base_observer.reset(ego_vehicle_simulation)
