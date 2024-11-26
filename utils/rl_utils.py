import torch
import os
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, VecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import numpy as np
import wandb
import torch

from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from commonroad_geometric.learning.reinforcement.experiment import RLExperiment, RLExperimentConfig
from utils.dataset_utils import EnvironmentDataset, get_data_dimensions
from utils.file_utils import find_model_path
from utils.training_utils import load_model_state, compute_rl_checksums
import wandb
from functools import partial
from models import get_model_class


EPS = 1e-5


class BaseWandbCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super(BaseWandbCallback, self).__init__(verbose)
        self.n_episodes = 0
        self.n_rollouts = 0
        self._info_buffer = {}

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        if not wandb.run:
            return
        self._info_buffer = {}
        self._log_policy_metrics()

    def _on_step(self) -> bool:
        if not wandb.run:
            return True
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
        if not wandb.run:
            return
        rollout_buffer = self.locals.get('rollout_buffer')
        if rollout_buffer is not None:
            try:
                self._log_rollout_metrics(rollout_buffer)
            except Exception as e:
                print(f"Error logging rollout metrics: {e}")
        self.n_rollouts += 1

    def _log_episode_metrics(self, rollout_buffer, n_episodes_done_step, n_steps, last_done_array, last_info):
        if not wandb.run:
            return
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

        self.n_episodes += n_episodes_done_step
        wandb.log({"info/n_episodes": float(self.n_episodes)}, step=self.num_timesteps)

    def _log_rollout_metrics(self, rollout_buffer):
        if not wandb.run:
            return
        for attr in ['advantages', 'values']:
            buffer_metrics = self._analyze_buffer_array(getattr(rollout_buffer, attr))
            for metric_name, value in buffer_metrics.items():
                if isinstance(value, np.ndarray):
                    for idx, idx_value in enumerate(value):
                        wandb.log({f"train/{attr}_ep_{metric_name}_idx_{idx}": idx_value}, step=self.num_timesteps)
                else:
                    wandb.log({f"train/{attr}_ep_{metric_name}": value}, step=self.num_timesteps)

    def _log_policy_metrics(self):
        if self.num_timesteps == 0 or not wandb.run:
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
                f"gradients/{name}_vanished": torch.mean((absgrad < 1e-5).float()).item(),
                f"weights/{name}_max": torch.max(weights).item(),
                f"weights/{name}_min": torch.min(weights).item(),
                f"weights/{name}_absmean": torch.mean(absweights).item(),
                f"weights/{name}_absmax": torch.max(absweights).item(),
                f"weights/{name}_dead": torch.mean((absweights < 1e-5).float()).item(),
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
        return BaseWandbCallback._analyze_buffer_array(masked_buffer)

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
    
# Custom callback to save the latest model
class LatestModelCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        
    def _init_callback(self) -> None:
        # Create save directory if it doesn't exist
        self.save_path.mkdir(parents=True, exist_ok=True)
        
    def _on_step(self) -> bool:
        if self.save_freq <= 0:
            return True
        if self.n_calls % self.save_freq == 0:
            latest_path = self.save_path / "latest_model.zip"
            self.model.save(latest_path)
            if self.verbose > 0:
                print(f"Saving latest model to {latest_path}")
                checksums = compute_rl_checksums(rl_model=self.model)
                print("Model checksums:", checksums)
        return True

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
                    self.model.save(os.path.join(self.best_model_save_path, "best_model.zip"))
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.best_model_save_path}")
                        checksums = compute_rl_checksums(rl_model=self.model)
                        print("Model checksums:", checksums)
                self.best_mean_reward = mean_reward

            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    "eval/mean_reward": mean_reward,
                    "eval/mean_ep_length": mean_ep_length,
                }, step=self.num_timesteps)

        return True

def create_representation_model(cfg, device, load=True, eval=True):
    """
    Create and load a representation model based on configuration.

    Args:
        cfg (DictConfig): Configuration object.
        device (torch.device): Device to load the model on.
        load (bool): Whether to load the model weights.
        eval (bool): Whether to set the model to evaluation mode.
    """
    model_save_dir = Path(cfg['project_dir']) / "models"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Load the full dataset for dimensions
    full_dataset = EnvironmentDataset(cfg)
    obs_shape, action_dim, ego_state_dim = get_data_dimensions(full_dataset)
    obs_shape = (cfg.dataset.t_pred, 3, cfg.viewer.window_size, cfg.viewer.window_size)

    # Initialize model
    ModelClass = get_model_class(cfg['representation']['model_type'])
    if ModelClass is None:
        raise ValueError(f"Invalid model type: {cfg['representation']['model_type']}")

    model = ModelClass(
        obs_shape=obs_shape, 
        action_dim=action_dim, 
        ego_state_dim=ego_state_dim,
        cfg=cfg,
        eval_mode=False
    )

    if load:
        # Find and load model weights
        model_path = find_model_path(cfg['project_dir'], cfg['representation']['model_path'])
        if model_path is None:
            raise FileNotFoundError(
                f"Model file not found: {cfg['representation']['model_path']}. "
                f"Searched in {cfg['project_dir']} and its subdirectories."
            )
        
        load_model_state(model_path, model, device)
        print(f"Loaded {ModelClass.__name__} model from: {model_path}")
    else:
        print(f"Created new {ModelClass.__name__} model")

    model.to(device)
    if eval:
        model.eval()


    return model


