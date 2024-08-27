import torch
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from experiment_setup import setup_experiment, create_rl_experiment_config
from models.predictive_model_v8 import PredictiveModelV8  # Or whichever model you used
from stable_baselines3 import PPO
import os
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from utils.dataset_utils import EnvironmentDataset, get_data_dimensions, create_data_loaders, move_batch_to_device
import numpy as np
import wandb

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

class RepresentationObserver:
    def __init__(self, representation_model, device, debug=False):
        self.representation_model = representation_model
        self.device = device
        self.representation_model.eval()
        self.debug = debug

    def __call__(self, obs):
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            if len(obs_tensor.shape) == 3:  # If single observation
                obs_tensor = obs_tensor.unsqueeze(0)
            rep = self.representation_model.encode(obs_tensor)
        if self.debug:
            print(f"Observation shape: {obs.shape}, Representation shape: {rep.shape}")
        return rep.cpu().numpy()

def create_representation_model(cfg):
    dataset_path = Path(cfg.project_dir) / "dataset"
    model_save_dir = Path(cfg.project_dir) / "models"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Load the full dataset
    full_dataset = EnvironmentDataset(dataset_path, downsample_factor=cfg.training.downsample_factor)

    # Get data dimensions
    obs_shape, action_dim, ego_state_dim = get_data_dimensions(full_dataset)

    model = PredictiveModelV8(obs_shape=obs_shape, action_dim=action_dim, ego_state_dim=ego_state_dim)
    model_path = Path(cfg.project_dir) / cfg.representation.model_path
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    return model

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Initialize wandb if enabled
    if cfg.wandb.enabled:
        wandb.init(project=cfg.wandb.project, config=OmegaConf.to_container(cfg, resolve=True))

    # Setup the experiment and environment
    experiment, env = setup_experiment(OmegaConf.to_container(cfg, resolve=True))

    representation_model = create_representation_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representation_model.to(device)

    # Create a new observer that uses the representation model
    representation_observer = RepresentationObserver(representation_model, device, debug=cfg.debug_mode)

    # Modify the environment to use the new observer
    rl_experiment_config = create_rl_experiment_config(OmegaConf.to_container(cfg, resolve=True))
    rl_experiment_config.env_options['observer'] = representation_observer

    # Recreate the environment with the new config
    env = experiment.make_env(
        scenario_dir=cfg.scenario_dir,
        n_envs=cfg.rl_training.num_envs,
        seed=cfg.seed,
        config=rl_experiment_config
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
        config=rl_experiment_config
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