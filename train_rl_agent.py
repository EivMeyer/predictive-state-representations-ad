import torch
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from experiment_setup import setup_experiment, create_rl_experiment_config, create_render_observer
from models.predictive_model_v8 import PredictiveModelV8  # Or whichever model you used
from commonroad_geometric.learning.reinforcement.experiment import RLExperiment, RLExperimentConfig
from stable_baselines3 import PPO
import os
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from utils.dataset_utils import EnvironmentDataset, get_data_dimensions, create_data_loaders, move_batch_to_device
import numpy as np
import wandb
import gymnasium
import numpy as np
import torch
from typing import Optional
from collections import deque

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.observer.base_observer import BaseObserver, T_Observation
from commonroad_geometric.learning.reinforcement.observer.implementations.render_observer import RenderObserver
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer, TrafficSceneRendererOptions


class VideoRecorderEvalCallback(EvalCallback):
    def __init__(self, eval_env, video_folder, video_freq, video_length, *args, **kwargs):
        super().__init__(eval_env, *args, **kwargs)
        self.video_folder = video_folder
        self.video_freq = video_freq
        self.video_length = video_length
        self.video_recorder = None

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Create a new video recorder for each evaluation
            self.video_recorder = VecVideoRecorder(
                self.eval_env,
                self.video_folder,
                record_video_trigger=lambda x: x == 0,  # Start recording immediately
                video_length=self.video_length,
                name_prefix=f"eval-{self.n_calls}"
            )
            
            # Evaluate the agent and record video
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.video_recorder,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
            )

            # Close the video recorder
            self.video_recorder.close()

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
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward

            # Log to wandb
            if wandb.run is not None:
                video_path = os.path.join(self.video_folder, f"eval-{self.n_calls}-step-0-to-step-{self.video_length}.mp4")
                wandb.log({
                    "eval/mean_reward": mean_reward,
                    "eval/mean_ep_length": mean_ep_length,
                    "eval/video": wandb.Video(video_path, fps=30, format="mp4")
                }, step=self.num_timesteps)

        return True

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
    def __init__(self, representation_model, device, render_observer: RenderObserver, sequence_length: int, debug=False):
        super().__init__()
        self.representation_model = representation_model
        self.device = device
        self.render_observer = render_observer
        self.sequence_length = sequence_length
        self.debug = debug
        self.representation_model.eval()
        from collections import deque
        self.obs_buffer = deque(maxlen=sequence_length)
        self.ego_state_buffer = deque(maxlen=sequence_length)
        self.is_first_observation = True

    def setup(self, dummy_data: Optional[CommonRoadData] = None) -> gymnasium.Space:
        render_space = self.render_observer.setup(dummy_data)
        
        dummy_obs = np.zeros((self.sequence_length, *render_space.shape), dtype=np.float32)
        dummy_ego_state = np.zeros((self.sequence_length, 4), dtype=np.float32)  # Assuming 4 ego state variables
        
        with torch.no_grad():
            dummy_obs_tensor = torch.from_numpy(dummy_obs).float().unsqueeze(0).permute(0, 1, 4, 2, 3).to(self.device)
            dummy_ego_state_tensor = torch.from_numpy(dummy_ego_state).float().unsqueeze(0).to(self.device)
            dummy_rep = self.representation_model.encode(dummy_obs_tensor, dummy_ego_state_tensor)
        
        rep_shape = dummy_rep.cpu().numpy().shape[1:]
        
        return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=rep_shape, dtype=np.float32)

    def observe(
        self,
        data: CommonRoadData,
        ego_vehicle_simulation: EgoVehicleSimulation
    ) -> T_Observation:
        render_obs = self.render_observer.observe(data, ego_vehicle_simulation)
        
        ego_state = np.array([
            ego_vehicle_simulation.ego_vehicle.state.velocity,
            0.0,
            ego_vehicle_simulation.ego_vehicle.state.steering_angle,
            ego_vehicle_simulation.ego_vehicle.state.yaw_rate
        ])
        
        self.obs_buffer.append(render_obs)
        self.ego_state_buffer.append(ego_state)
        
        # If it's the first observation of an episode, fill the buffer with the current observation
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
            print(f"Observation sequence shape: {obs_sequence.shape}")
            print(f"Ego state sequence shape: {ego_state_sequence.shape}")
            print(f"Representation shape: {representation.shape}")
        
        return representation

    def reset(self) -> None:
        """Reset the observer when a new episode starts."""
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

    model = PredictiveModelV8(obs_shape=obs_shape, action_dim=action_dim, ego_state_dim=ego_state_dim, num_frames_to_predict=cfg.dataset.t_pred)
    model_path = Path(cfg.project_dir) / cfg.representation.model_path
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    return model

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Initialize wandb if enabled
    if cfg.wandb.enabled:
        wandb.init(project=cfg.wandb.project, config=OmegaConf.to_container(cfg, resolve=True))


    representation_model = create_representation_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representation_model.to(device)

    render_observer = create_render_observer(OmegaConf.to_container(cfg, resolve=True)['viewer'])

    # Create a new observer that uses the representation model
    representation_observer = RepresentationObserver(representation_model, device, debug=cfg.debug_mode, render_observer=render_observer, sequence_length=cfg.dataset.t_obs)

    """Set up the RL experiment using the provided configuration."""
    rl_experiment_config = create_rl_experiment_config(OmegaConf.to_container(cfg, resolve=True))
    rl_experiment_config.env_options['observer'] = representation_observer

    experiment = RLExperiment(config=rl_experiment_config)

    # Create the environment
    env = experiment.make_env(
        scenario_dir=Path(cfg.scenario_dir),
        n_envs=cfg.dataset.num_workers,
        seed=cfg.seed
    )

    # Normalize the environment
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Create the RL agent
    model = PPO("MlpPolicy", env, verbose=1, device=device,
                learning_rate=cfg.rl_training.learning_rate,
                n_steps=cfg.rl_training.n_steps,
                batch_size=cfg.rl_training.batch_size,
                n_epochs=cfg.rl_training.n_epochs,
                gamma=cfg.rl_training.gamma,
                gae_lambda=cfg.rl_training.gae_lambda,
                clip_range=cfg.rl_training.clip_range,
                ent_coef=cfg.rl_training.ent_coef,
                vf_coef=cfg.rl_training.vf_coef,
                max_grad_norm=cfg.rl_training.max_grad_norm,
                tensorboard_log=Path(cfg.project_dir) / "tensorboard_logs" if cfg.wandb.enabled else None)

    # Setup evaluation environment
    eval_env = experiment.make_env(
        scenario_dir=cfg.scenario_dir,
        n_envs=1,
        seed=cfg.seed + 1000,  # Different seed for eval env
        observer=representation_observer
    )
    
    # Use the same normalization stats as the training environment
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms
    
    # Setup video recording
    video_folder = Path(cfg.project_dir) / cfg.rl_training.video_folder
    video_folder.mkdir(parents=True, exist_ok=True)

    # Setup evaluation callback with video recording
    eval_callback = VideoRecorderEvalCallback(
        eval_env=eval_env,
        video_folder=str(video_folder),
        video_freq=cfg.rl_training.video_freq,
        video_length=cfg.rl_training.video_length,
        best_model_save_path=Path(cfg.project_dir) / cfg.rl_training.save_path,
        log_path=Path(cfg.project_dir) / cfg.rl_training.log_path,
        eval_freq=cfg.rl_training.eval_freq,
        deterministic=True,
        render=False
    )

    # Setup callbacks
    callbacks = [eval_callback]
    if cfg.debug_mode:
        debug_callback = DebugCallback()
        callbacks.append(debug_callback)

    # Train the agent
    model.learn(total_timesteps=cfg.rl_training.total_timesteps, callback=callbacks,
                tb_log_name="ppo_run" if cfg.wandb.enabled else None)

    # Save the final model
    final_model_path = Path(cfg.project_dir) / cfg.rl_training.save_path / "final_model"
    model.save(final_model_path)

    if cfg.wandb.enabled:
        wandb.save(str(final_model_path))
        wandb.finish()

if __name__ == "__main__":
    main()