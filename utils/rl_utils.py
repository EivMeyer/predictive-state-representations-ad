import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from experiment_setup import setup_base_experiment, create_base_experiment_config, create_render_observer
from commonroad_geometric.learning.reinforcement.experiment import RLExperiment, RLExperimentConfig
import os
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, VecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from utils.dataset_utils import EnvironmentDataset, get_data_dimensions
from utils.file_utils import find_model_path
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



import numpy as np
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback

EPS = 1e-5

class WandbCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super(WandbCallback, self).__init__(verbose)
        self.n_episodes = 0
        self.n_rollouts = 0
        self._info_buffer = {}

    def _on_training_start(self) -> None:
        self._termination_reasons = self.training_env.get_attr('termination_reasons')[0]

    def _on_rollout_start(self) -> None:
        self._info_buffer = {}
        self._log_policy_metrics()

    def _on_step(self) -> bool:
        assert self.logger is not None

        n_steps = self.locals.get('n_steps')
        if n_steps is None:
            return True
        
        rollout_buffer = self.locals.get('rollout_buffer')
        infos = self.locals.get('infos')

        last_info = self._info_buffer.get(n_steps - 1, None)
        last_done_array = np.array([info['done'] for info in last_info]) if last_info is not None else None

        n_episodes_done_step = np.sum(last_done_array).item() if last_done_array is not None else 0
        if n_episodes_done_step > 0:
            self._log_episode_metrics(rollout_buffer, n_episodes_done_step, n_steps, last_done_array, last_info)

        self._info_buffer[n_steps] = infos
        self._info_buffer = {step: v for step, v in self._info_buffer.items() if step >= n_steps - 5}

        return True

    def _on_rollout_end(self) -> None:
        rollout_buffer = self.locals.get('rollout_buffer')
        if rollout_buffer is not None:
            self._log_rollout_metrics(rollout_buffer)
        self.n_rollouts += 1

    def _log_episode_metrics(self, rollout_buffer, n_episodes_done_step, n_steps, last_done_array, last_info):
        for attr in ['actions', 'log_probs', 'rewards', 'values']:
            buffer_metrics = self._analyze_buffer_array_masked(
                buffer=getattr(rollout_buffer, attr),
                n_steps=n_steps,
                done_array=last_done_array
            )
            for metric_name, value in buffer_metrics.items():
                if isinstance(value, np.ndarray):
                    for idx, idx_value in enumerate(value):
                        wandb.log({f"train/{attr}_ep_{metric_name}_idx_{idx}": float(idx_value)}, step=self.num_timesteps)
                else:
                    wandb.log({f"train/{attr}_ep_{metric_name}": float(value)}, step=self.num_timesteps)

        termination_criteria_flags = dict.fromkeys(self._termination_reasons, False)
        done_indices = np.where(last_done_array)[0]
        for env_index in done_indices:
            env_info = last_info[env_index]
            termination_reason = env_info.get('termination_reason')
            termination_criteria_flags[termination_reason] = True

            env_reward_component_info = env_info['reward_component_episode_info']
            for reward_component, component_info in env_reward_component_info.items():
                for component_metric, component_value in component_info.items():
                    wandb.log({f"train/reward_{reward_component}_ep_{component_metric}": float(component_value)}, step=self.num_timesteps)

            vehicle_aggregate_stats = env_info['vehicle_aggregate_stats']
            for state, state_info in vehicle_aggregate_stats.items():
                for state_metric, state_value in state_info.items():
                    wandb.log({f"train/vehicle_{state}_ep_{state_metric}": float(state_value)}, step=self.num_timesteps)

            wandb.log({
                "train/ep_num_obstacles": float(env_info.get('total_num_obstacles', 0)),
                "train/ep_cumulative_reward": float(env_info.get('cumulative_reward', 0)),
                "train/next_reset_ready": float(env_info.get('next_reset_ready', False)),
                "train/ep_length": float(env_info.get('time_step', 0))
            }, step=self.num_timesteps)

        for termination_criteria in self._termination_reasons:
            wandb.log({
                f"train/termination_{termination_criteria}": 
                float(termination_criteria_flags[termination_criteria])
            }, step=self.num_timesteps)

        self.n_episodes += n_episodes_done_step
        wandb.log({"info/n_episodes": float(self.n_episodes)}, step=self.num_timesteps)


    def _log_rollout_metrics(self, rollout_buffer):
        for attr in ['advantages', 'values']:
            buffer_metrics = self._analyze_buffer_array(getattr(rollout_buffer, attr))
            for metric_name, value in buffer_metrics.items():
                if isinstance(value, np.ndarray):
                    for idx, idx_value in enumerate(value):
                        wandb.log({f"train/{attr}_ep_{metric_name}_idx_{idx}": idx_value}, step=self.num_timesteps)
                else:
                    wandb.log({f"train/{attr}_ep_{metric_name}": value}, step=self.num_timesteps)

    def _log_policy_metrics(self):
        if self.num_timesteps == 0:
            return

        for name, param in self.model.policy.named_parameters():
            grad = param._grad
            weights = param.data
            if grad is None:
                continue
            absweights = torch.abs(weights)
            absgrad = torch.abs(grad)

            metrics = {
                f"gradients/{name}_max": torch.max(grad).item(),
                f"gradients/{name}_min": torch.min(grad).item(),
                f"gradients/{name}_absmean": torch.mean(absgrad).item(),
                f"gradients/{name}_absmax": torch.max(absgrad).item(),
                f"gradients/{name}_vanished": torch.mean((absgrad < EPS).float()).item(),
                f"weights/{name}_max": torch.max(weights).item(),
                f"weights/{name}_min": torch.min(weights).item(),
                f"weights/{name}_absmean": torch.mean(absweights).item(),
                f"weights/{name}_absmax": torch.max(absweights).item(),
                f"weights/{name}_dead": torch.mean((absweights < EPS).float()).item(),
            }

            if grad.numel() > 1:
                metrics[f"gradients/{name}_std"] = torch.std(grad).item()
            if weights.numel() > 1:
                metrics[f"weights/{name}_std"] = torch.std(weights).item()

            wandb.log(metrics, step=self.num_timesteps)

        wandb.log({"info/n_rollouts": self.n_rollouts}, step=self.num_timesteps)

    @staticmethod
    def _analyze_buffer_array_masked(buffer: np.ndarray, n_steps: int, done_array: np.ndarray) -> dict:
        masked_buffer = buffer[:n_steps, done_array, ...]
        return WandbCallback._analyze_buffer_array(masked_buffer)

    @staticmethod
    def _analyze_buffer_array(buffer: np.ndarray) -> dict:
        return {
            "max": np.max(buffer, axis=(0, 1)),
            "min": np.min(buffer, axis=(0, 1)),
            "std": np.std(buffer, axis=(0, 1)),
            "mean": np.mean(buffer, axis=(0, 1)),
            "absmean": np.mean(np.abs(buffer), axis=(0, 1))
        }


class VecEnvWrapperWithReset(VecEnvWrapper):
    def reset(self):
        return self.venv.reset()

class DebugCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(DebugCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        self.rewards.append(self.locals['rewards'])
        if self.n_calls % 1 == 0: 
            avg_reward = np.mean(self.rewards[-1:])
            print(f"Step {self.n_calls}, Reward: {avg_reward:.2f}")
        return True


class RepresentationObserver(BaseObserver):
    def __init__(
        self, 
        representation_model, 
        device, 
        render_observer: RenderObserver, 
        sequence_length: int, 
        include_ego_state: bool = True,
        debug: bool = False
    ):
        super().__init__()
        self.representation_model = representation_model
        self.device = device
        self.render_observer = render_observer
        self.sequence_length = sequence_length
        self.include_ego_state = include_ego_state
        self.debug = debug
        self.representation_model.eval()
        self.obs_buffer = deque(maxlen=sequence_length)
        self.ego_state_buffer = deque(maxlen=sequence_length)
        self.is_first_observation = True
        
        if self.debug:
            self.setup_debug_plot()

    def setup_debug_plot(self):
        self.fig = plt.figure(figsize=(20, 6))
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 2, 1])

        # Current observation
        self.ax_obs = self.fig.add_subplot(gs[0, 0])
        self.im_obs = self.ax_obs.imshow(np.zeros((64, 64, 3)))
        self.ax_obs.set_title('Current Observation', fontsize=12, pad=10)
        self.ax_obs.axis('off')

        # Predictions grid
        self.ax_pred = self.fig.add_subplot(gs[0, 1])
        self.ax_pred.axis('off')
        self.im_preds = []

        for i in range(3):
            for j in range(3):
                # Adjust inset axes positions and ensure adequate spacing
                ax = self.ax_pred.inset_axes([j/3 + 0.01, (2-i)/3 + 0.04, 0.28, 0.28])  # Reduced size and shifted positions
                im = ax.imshow(np.zeros((64, 64, 3)))
                self.im_preds.append(im)
                ax.axis('off')
                step = i * 3 + j
                label = 't' if step == 0 else f't+{step}'
                ax.set_title(label, fontsize=10, pad=8)  # Increased pad to separate labels from images

        self.ax_pred.set_title('Predictions', fontsize=14, pad=20)  # Increased pad for main title to avoid overlap with top row

        # Latent representation
        self.ax_rep = self.fig.add_subplot(gs[0, 2])
        self.im_rep = self.ax_rep.imshow(np.zeros((1, 1)), cmap='viridis', aspect='equal')
        self.ax_rep.set_title('Latent Representation', fontsize=12, pad=10)
        self.fig.colorbar(self.im_rep, ax=self.ax_rep)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to reduce space taken by tight_layout
        plt.ion()
        plt.show()

    def setup(self, dummy_data: Optional[CommonRoadData] = None) -> gymnasium.Space:
        render_space = self.render_observer.setup(dummy_data)

        dummy_obs = np.zeros((self.sequence_length, *render_space.shape), dtype=np.float32)
        dummy_ego_state = np.zeros((self.sequence_length, 4), dtype=np.float32)  # Assuming 4 ego state variables

        with torch.no_grad():
            dummy_obs_tensor = torch.from_numpy(dummy_obs).float().unsqueeze(0).permute(0, 1, 4, 2, 3).to(self.device)
            dummy_ego_state_tensor = torch.from_numpy(dummy_ego_state).float().unsqueeze(0).to(self.device)
            dummy_rep = self.representation_model.encode(dummy_obs_tensor, dummy_ego_state_tensor)

        rep_shape = dummy_rep.cpu().numpy().shape[1:]

        obs_shape = rep_shape if not self.include_ego_state else (rep_shape[0] + dummy_ego_state.shape[-1],)

        return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

    def observe(
        self,
        data: CommonRoadData,
        ego_vehicle_simulation: EgoVehicleSimulation
    ) -> T_Observation:
        render_obs = self.render_observer.observe(data, ego_vehicle_simulation) / 255.0

        ego_state = np.array([
            ego_vehicle_simulation.ego_vehicle.state.velocity,
            0.0,
            ego_vehicle_simulation.ego_vehicle.state.steering_angle,
            ego_vehicle_simulation.ego_vehicle.state.yaw_rate
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
            rep = self.representation_model.encode(obs_tensor, ego_state_tensor)
            
        representation = rep.cpu().numpy().squeeze()

        if self.debug:
            with torch.no_grad():
                decoding = self.representation_model.decode(rep)
            predictions = decoding[0].permute(0, 2, 3, 1).cpu().detach().numpy()

            self.update_debug_plot(render_obs, predictions, representation)

        if self.include_ego_state:
            representation = np.concatenate([representation, ego_state])

        return representation

    def update_debug_plot(self, current_obs, predictions, representation):
        # Update current observation
        self.im_obs.set_data(current_obs)

        # Update predictions
        for i, im in enumerate(self.im_preds):
            if i < len(predictions):
                # Ensure the prediction is in the same orientation as the current observation
                pred = predictions[i]
                im.set_data(pred)
            else:
                im.set_data(np.zeros_like(current_obs))

            # Ensure aspect ratio matches the current observation
            im.set_extent([0, current_obs.shape[1], current_obs.shape[0], 0])

        # Update latent representation
        rep_dim = int(np.sqrt(representation.shape[0]))
        rep_reshaped = representation.reshape(rep_dim, rep_dim)
        self.im_rep.set_data(rep_reshaped)
        self.im_rep.autoscale()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def reset(self, ego_vehicle_simulation: EgoVehicleSimulation) -> None:
        self.obs_buffer.clear()
        self.ego_state_buffer.clear()
        self.is_first_observation = True
        if self.debug:
            print("RepresentationObserver reset: Cleared observation and ego state buffers.")

def create_representation_model(cfg):
    dataset_path = Path(cfg.project_dir) / "dataset"
    model_save_dir = Path(cfg.project_dir) / "models"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Load the full dataset
    full_dataset = EnvironmentDataset(dataset_path, downsample_factor=cfg.training.downsample_factor)

    # Get data dimensions
    obs_shape, action_dim, ego_state_dim = get_data_dimensions(full_dataset)

    # Get the model class based on the config
    ModelClass = get_model_class(cfg.training.model_type)
    if ModelClass is None:
        raise ValueError(f"Invalid model type: {cfg.training.model_type}")

    model = ModelClass(obs_shape=obs_shape, action_dim=action_dim, ego_state_dim=ego_state_dim, num_frames_to_predict=cfg.dataset.t_pred, hidden_dim=cfg.training.hidden_dim)

    # Find the correct model path
    model_path = find_model_path(cfg.project_dir, cfg.representation.model_path)
    if model_path is None:
        raise FileNotFoundError(f"Model file not found: {cfg.representation.model_path}. "
                                f"Searched in {cfg.project_dir} and its subdirectories.")
    
    print(f"Using model file: {model_path}")

    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    return model

def has_reached_end(simulation: EgoVehicleSimulation, arclength_threshold: float) -> bool:
    ego_position = simulation.ego_vehicle.state.position
    ego_trajectory_polyline = simulation.ego_vehicle.ego_route.planning_problem_path_polyline
    arclength = ego_trajectory_polyline.get_projected_arclength(
        ego_position,
        relative=True,
        linear_projection=True
    )
    reached_end = arclength >= arclength_threshold
    return reached_end

class VideoRecorderEvalCallback(EvalCallback):
    def __init__(self, eval_env, video_folder, video_freq, video_length, n_eval_episodes, *args, **kwargs):
        super().__init__(eval_env, *args, **kwargs)
        self.video_folder = video_folder
        self.video_freq = video_freq
        self.video_length = video_length
        self.video_recorder = None
        self.n_eval_episodes = n_eval_episodes
        self.last_video_step = 0

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Determine if this evaluation should be recorded
            should_record = (self.n_calls - self.last_video_step) >= self.video_freq

            if should_record:
                # Create a new video recorder for this evaluation
                self.video_recorder = VecVideoRecorder(
                    self.eval_env,
                    self.video_folder,
                    record_video_trigger=lambda x: x == 0,  # Start recording immediately
                    video_length=self.video_length,
                    name_prefix=f"eval-{self.n_calls}"
                )
                self.last_video_step = self.n_calls

            # Evaluate the agent
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.video_recorder if should_record else self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
            )

            if should_record:
                # Close the video recorder
                self.video_recorder.close()

                # Log to wandb
                if wandb.run is not None:
                    video_path = os.path.join(self.video_folder, f"eval-{self.n_calls}-step-0-to-step-{self.video_length}.mp4")
                    wandb.log({
                        "eval/video": wandb.Video(video_path, fps=30, format="mp4")
                    }, step=self.num_timesteps)

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward

            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    "eval/mean_reward": mean_reward,
                    "eval/mean_ep_length": mean_ep_length,
                }, step=self.num_timesteps)

        return True
class CustomReachedEndCriterion(BaseTerminationCriterion):
    def __init__(self, arclength_threshold: float):
        self.arclength_threshold = arclength_threshold
        super().__init__()

    def __call__(
        self,
        simulation: EgoVehicleSimulation
    ) -> Tuple[bool, Optional[str]]:
        reached_end = has_reached_end(simulation, self.arclength_threshold)
        return reached_end, 'ReachedEnd' if reached_end else None

    @property
    def reasons(self) -> Set[str]:
        return {'ReachedEnd'}

class CustomReachedEndRewardComputer(BaseRewardComputer):
    def __init__(self, arclength_threshold: float, reward: float):
        self.arclength_threshold = arclength_threshold
        self.reward = reward
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData,
        observation: T_Observation
    ) -> float:
        reached_end = has_reached_end(simulation, self.arclength_threshold)
        if reached_end:
            return self.reward
        return 0.0


def create_termination_criteria():
    termination_criteria = [
        CollisionCriterion(),
        OffroadCriterion(),
        # ReachedGoalCriterion(),
        # OvershotGoalCriterion(),
        TimeoutCriterion(max_timesteps=500),
        CustomReachedEndCriterion(arclength_threshold=1.0)
    ]
    return termination_criteria


def create_rewarders():
    rewarders = [
        # AccelerationPenaltyRewardComputer(
        #     weight=0.0,
        #     loss_type=RewardLossMetric.L2
        # ),
        CustomReachedEndRewardComputer(
            arclength_threshold=1.0,
            reward=3.5
        ),
        CollisionPenaltyRewardComputer(
            penalty=-2.0,
        ),
        # FrictionViolationPenaltyRewardComputer(penalty=-0.01),
        TrajectoryProgressionRewardComputer(
            weight=0.2,
            delta_threshold=0.08
        ),
        ConstantRewardComputer(reward=-0.001),
        #
        # ReachedGoalRewardComputer(reward=3.5),
        # OvershotGoalRewardComputer(reward=0.0),
        # SteeringAnglePenaltyRewardComputer(weight=0.0005, loss_type=RewardLossMetric.L1),
        StillStandingPenaltyRewardComputer(penalty=-0.05, velocity_threshold=2.0),
        # TimeToCollisionPenaltyRewardComputer(weight=0.1), # requires incoming edges
        OffroadPenaltyRewardComputer(penalty=-3.5),
        VelocityPenaltyRewardComputer(
            reference_velocity=56.0,
            weight=0.002,
            loss_type=RewardLossMetric.L2,
            only_upper=True
        ),

        # LateralErrorPenaltyRewardComputer(weight=0.0001, loss_type=RewardLossMetric.L1),
        YawratePenaltyRewardComputer(weight=0.01),
        # HeadingErrorPenaltyRewardComputer(
        #     weight=0.01,
        #     loss_type=RewardLossMetric.L2,
        #     wrong_direction_penalty=-0.01
        # )
    ]

    return rewarders

def create_representation_observer(cfg):
    representation_model = create_representation_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representation_model.to(device)
    render_observer = create_render_observer(OmegaConf.to_container(cfg, resolve=True)['viewer'])
    representation_observer = RepresentationObserver(representation_model, device, debug=cfg.debug_mode, render_observer=render_observer, sequence_length=cfg.dataset.t_obs)
    return representation_observer

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
    experiment_config.respawner_options['init_speed'] = 'auto'
    experiment_config.control_space_options['lower_bound_acceleration'] = -10.0
    experiment_config.control_space_options['upper_bound_acceleration'] = 10.0
    experiment_config.rewarder = SumRewardAggregator(create_rewarders())
    experiment_config.termination_criteria = create_termination_criteria()

    experiment = RLExperiment(config=experiment_config)

    return experiment